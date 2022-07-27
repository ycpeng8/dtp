import math 
from collections import OrderedDict
import random
from typing import Dict, List 
import hydra
import neptune.new as neptune 
# from comet_ml import Experiment 
import numpy as np
from numpy.testing._private.utils import runstring
import torch
from omegaconf import DictConfig
from torch import nn, optim, cuda 
from torch.nn.functional import one_hot
import pytorch_lightning as pl 
# from torch.nn.modules.loss import BCEWithLogitsLoss
import os 

import datamodules 
import utils 


# init random seeds python, numpy, pytorch
# pl.seed_everything(10000)
# DEVICE = torch.device("cuda:0" if cuda.is_available() else "cpu")


class SimpleRNN(nn.Module):
    def __init__(self, 
                method: str, 
                in_dim: int,
                hidden_dim: int,
                out_dim: int,
                actv_fn: str,
                update_g_only_before: int, 
                device=torch.device('cpu') 
                ):
        
        super().__init__()
        
        self.method = method
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.actv_fn = actv_fn 
        self.update_g_only_before = update_g_only_before
        self.opt_params = None 
        self.optimizers = None 
        self.device = device 

        if self.actv_fn == 'tanh':
            self.act = nn.Tanh()
        if self.actv_fn == 'relu':
            self.act = nn.ReLU()

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        self.optimizers = None # to be initialized by set_optimizers()
        self.logger = None # set in fit()

        # trainable parameters 
        self.W_hh = nn.Parameter(
            utils.rand_ortho(
                (self.hidden_dim, self.hidden_dim),
                np.sqrt(6. / (self.hidden_dim + self.hidden_dim))))
        self.W_xh = nn.Parameter(torch.empty(self.in_dim, self.hidden_dim))
        nn.init.normal_(self.W_xh, 0, 0.1)
        self.W_hy = nn.Parameter(torch.empty(self.hidden_dim, self.out_dim))
        nn.init.normal_(self.W_hy, 0, 0.1)
        self.b_h = nn.Parameter(torch.zeros([self.hidden_dim]))
        self.b_y = nn.Parameter(torch.zeros([self.out_dim]))

        # params for inverse functions 
        if self.method == 'targetprop':
            self.V_hh = nn.Parameter(utils.rand_ortho(
                (self.hidden_dim, self.hidden_dim),
                np.sqrt(6. / (self.hidden_dim + self.hidden_dim))))
            self.c_h = nn.Parameter(torch.zeros([self.hidden_dim]))

    def f_func(self, h, x): 
        '''
        Step forward function. One step forward in time, keep backward gradient of parameters
        '''
        return self.act(h @ self.W_hh + x @ self.W_xh + self.b_h)

    def y_func(self, h):
        '''
        Hidden state to output. Outputs are logits and not normalized 
        '''
        return h @ self.W_hy + self.b_y

    def g_func(self, hp1, xp1):
        '''
        Given h_{t+1} and x_{t+1}, calculate h_{t}. \\
        self.W_xh is not trainable in backward step using detach()
        '''
        assert (self.method=='targetprop')
        return self.act(hp1 @ self.V_hh + xp1 @ self.W_xh.detach() + self.c_h)

    def forward(self, X):
        '''
        forward() is used for inference only, independent from learning procedure defined in training_step \\
        X: input batch if size [batch_size, seq_len, vocab_size] \\
        return value: output logits of size [batch_size, seq_len, vocab_size] 
        '''
        X = X.to(self.device)
        # print(X.dtype) # 
        X = one_hot(X.transpose(0, 1)).float() # seq_len * batch_size * vocab_size
        seq_len, batch_size, vocab_size = X.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        out_logits = []
        for time_idx in range(seq_len):
            x_t = X[time_idx, :, :]
            h = self.f_func(h, x_t) # post activation 
            out_logits.append(self.y_func(h))
        return torch.stack(out_logits, dim=0).squeeze_().transpose(0, 1) 

    def set_optimizers(self, opt_params):
        '''
        Set the optimizers in model \\
        OPTIMIZERS: {'opt_bp': Adam()} or {'opt_f': Adam(), 'opt_g': Adam()}
        '''
        if self.method == 'backprop':
            self.opt_params = opt_params 
            self.optimizers = {'opt_bp': optim.Adam(self.parameters(), lr=self.opt_params['opt_bp']['lr'])} 
        if self.method == 'targetprop':
            self.opt_params = opt_params  # including "lr_i" 
            self.optimizers = {
                'opt_g': optim.Adam(
                    [self.V_hh, self.c_h], 
                    lr=self.opt_params['opt_g']['lr'],
                    betas=(self.opt_params['opt_g']['beta1'], self.opt_params['opt_g']['beta2'])
                    ),
                'opt_f': optim.Adam(
                    [self.W_xh, self.W_hh, self.b_h, self.W_hy, self.b_y],
                    lr=self.opt_params['opt_f']['lr'],
                    betas=(self.opt_params['opt_f']['beta1'], self.opt_params['opt_f']['beta2'])
                )
            }

    # def if_update_g_only(self, batch_idx):
    #     # if batch_idx < 1000 or math.floor(batch_idx / 10) % 10 == 0:
    #     if batch_idx < 1000:
    #         return True
    #     else:
    #         return False 

    def training_step(self, batch, batch_idx):
        '''
        Main algorithmic step, implimenting backprop and targetprop depending on self.method \\
        BATCH: batch_sz * seq_len * vocab_size, torch.long \\
        BATCH_IDX: int 
        '''        
        batch_X, batch_Y = batch
        assert (batch_Y.dtype == torch.long)
        batch_X = batch_X.to(self.device)
        batch_Y = batch_Y.to(self.device)
        batch_size, seq_len = batch_Y.size()
        
        if self.method == 'backprop':
            loss = None 
            opt_bp = self.optimizers['opt_bp']
            ###
            opt_bp.zero_grad()
            logits = self(batch_X)
            loss = self.ce_loss(logits.permute(0, 2, 1), batch_Y)
            loss.backward()
            opt_bp.step()
            opt_bp.zero_grad()
            ###
            return {'bp_train_loss': loss} 
            
        elif self.method == 'targetprop':
            batch_X = one_hot(batch_X.transpose(0, 1)).float()
            if batch_idx < self.update_g_only_before:
            # if self.if_update_g_only(batch_idx):
                g_loss = torch.zeros(1, device=self.device)
                f_loss = torch.zeros(1, device=self.device)
                opt_g = self.optimizers['opt_g']
                ###
                opt_g.zero_grad()
                h_prev = torch.zeros([batch_size, self.hidden_dim], dtype=torch.float, device=self.device)
                # h_prev = None 
                for t in range(seq_len):
                    x_t = batch_X[t, :, :]
                    if t == 0:
                        with torch.no_grad():
                            h_prev = self.f_func(h_prev, x_t)
                    else:
                        with torch.no_grad():
                            h_t = self.f_func(h_prev, x_t)
                        rec_prev = self.g_func(h_t, x_t)
                        g_loss_t = self.mse_loss(h_prev, rec_prev)
                        g_loss += g_loss_t 
                        h_prev = h_t.detach()
                
                # for t in range(seq_len):
                #     # print(batch_X.shape) # 
                #     x_t = batch_X[t, :, :]
                #     # TODO: consider rec init hidden state 
                #     with torch.no_grad():
                #         h_t = self.f_func(h_prev, x_t)
                #     rec_prev = self.g_func(h_t, x_t)
                #     g_loss_t = self.mse_loss(h_prev, rec_prev)
                #     g_loss += g_loss_t 
                #     h_prev = h_t.detach()
                g_loss.backward() # training V_hh and c_h
                opt_g.step()
                ###
                return {'f_train_loss': f_loss, 'g_train_loss': g_loss}

            else: 
                f_loss = torch.zeros(1, device=self.device)
                g_loss = torch.zeros(1, device=self.device)
                opt_f = self.optimizers['opt_f']
                opt_g = self.optimizers['opt_g']
                ###
                opt_f.zero_grad()
                opt_g.zero_grad()

                target_init_step_size = self.opt_params['lr_i']
                h_t = torch.zeros([batch_size, self.hidden_dim], dtype=torch.float, device=self.device)
                Hs = [] 
                Ts = [] # local targets 

                ##### Step 1: forward pass to get hidden activations and local targets at each step #####
                recs = [] # rec of steps: 0 -> seq_len - 2
                for t in range(seq_len):
                    x_t = batch_X[t, :, :] # batch_size * vocab_size 
                    h_t = self.f_func(h_t.detach(), x_t) # trainable: W_hh, W_xh, b_h
                    Hs.append(h_t)
                    h_t_w_grad = h_t.detach().clone().requires_grad_()
                    y_pred_t = self.y_func(h_t_w_grad) # with grad: h_t_w_grad, W_hy, b_y
                    y_loss_t = self.ce_loss(y_pred_t, batch_Y[:, t])
                    y_loss_t.backward() # # accumulate W_hy and b_y grad, and set local target
                    with torch.no_grad():
                        target_t = h_t_w_grad - target_init_step_size * h_t_w_grad.grad 
                        Ts.append(target_t)
                    if t > 0: # compute g loss
                        rec_prev = self.g_func(h_t.detach(), x_t)
                        recs.append(rec_prev.detach().clone())
                        g_loss_t = self.mse_loss(rec_prev, Hs[-2].detach())
                        g_loss += g_loss_t 
                assert (not torch.all(self.W_hy.grad == 0)) # not all zeros
                assert (not torch.all(self.b_y.grad == 0)) # not all zeros 
                
                ##### Step 2: backward pass targets #####
                # backward pass to accumulate the targets with different target propagation
                # activations are in Hs list, and local targets are in Ts list, W_hy and b_y grad have been accumed
                combined_target_t_weighted = None
                for t in range(seq_len-1, -1, -1):
                    if t == seq_len - 1:
                        with torch.no_grad():
                            # all prev steps are contributing, thus scale down it 
                            # combined_target_t_weighted = Hs[t] + (Ts[t] - Hs[t])/(t+1.) # TODO 
                            combined_target_t_weighted = Ts[t] 
                    else: 
                        with torch.no_grad():
                            backproped_target = self.g_func(combined_target_t_weighted, batch_X[t+1, :, :])
                            linear_correction = Hs[t] - recs[t] # rec of steps 0 -> seq_len -2 
                            backproped_target_corrected = backproped_target + linear_correction 
                            # combined_target_t_weighted = backproped_target_corrected + (Ts[t]-Hs[t])/(t+1.) # TODO 
                            combined_target_t_weighted = backproped_target_corrected + (Ts[t]-Hs[t])
                    assert (combined_target_t_weighted.requires_grad == False)
                    assert (Hs[t].requires_grad == True)
                    f_loss += self.mse_loss(Hs[t], combined_target_t_weighted)

                g_loss.backward(retain_graph=True)
                opt_g.step() # update V_hh, c_h
                f_loss.backward()
                opt_f.step()
                
                opt_f.zero_grad()
                opt_g.zero_grad()                
                ###
                return {'f_train_loss': f_loss, 'g_train_loss': g_loss}
                
        else:
            print("Please set the methed of your model either to 'backproo' or 'targetprop'")
            exit() 
        
        

    def validate_with(self, val_loader):
        '''
        model.eval() and torch.no_grad() are called within fit() before and fater calling this function
        '''
        batch_val_losses = []
        batch_accs = []
        for batch_idx, (batch_X, batch_Y) in enumerate(val_loader):
            batch_X = batch_X.to(self.device)
            batch_Y = batch_Y.to(self.device)
            # batch_X = one_hot(batch_X.transpose(0, 1)).float()
            assert (batch_Y.dtype == torch.long)
            batch_size, seq_len = batch_Y.size()
            out_logits = self(batch_X) # one_hot is inside forward()
            correct_pred_count = utils.num_correct_samples(out_logits, batch_Y)
            batch_acc = correct_pred_count / batch_size 
            batch_val_loss = self.ce_loss(out_logits.permute(0, 2, 1), batch_Y)
            ###
            batch_val_losses.append(batch_val_loss)
            batch_accs.append(batch_acc)
            if self.fast_dev_run:
                break 

        # calculating avged metrics across whole val epoch
        val_loss = torch.stack(batch_val_losses).mean()
        val_acc = torch.stack(batch_accs).mean()
        return {'val_loss': val_loss, 'val_acc': val_acc}
                        

    def test_with(self, test_loader): 
        results = self.validate_with(test_loader)
        return {'test_loss': results['val_loss'], 'test_acc': results['val_acc']}

    def avg_results(self, results_list: List[Dict[str, torch.Tensor]])-> Dict[str, torch.Tensor]:
        num_steps = len(results_list)
        assert (num_steps > 0)
        accum_metrics = dict.fromkeys(results_list[0], torch.tensor([0.], device=self.device))
        # print(accum_metrics) 
        for result in results_list:
            for key in OrderedDict(result).keys():
                accum_metrics[key] = accum_metrics[key] + result[key] 
        # print(accum_metrics)
        for key in accum_metrics.keys():
            accum_metrics[key] = accum_metrics[key]/num_steps
        
        return accum_metrics

    def fit(self,
            train_loader,
            val_loader=None,
            val_every_n_steps=500,
            log_every_n_steps=10, # training metrics averaged over this number of steps
            fast_dev_run=False, # if True, no logger, run one batch of train, val
            logger=None
        ):
        train_results_list = []
        self.fast_dev_run = fast_dev_run
        for batch_idx, batch in enumerate(train_loader):
            train_step_result = self.training_step(batch, batch_idx)
            train_results_list.append(train_step_result)
            
            if ((batch_idx+1) % log_every_n_steps == 0 or batch_idx == 0 or self.fast_dev_run) and logger:
                avg_result = self.avg_results(train_results_list)
                for key, value in avg_result.items():
                    # print(value)
                    # logger.log_metric(key, batch_idx, value)
                    logger['metrics/'+key].log(value, step=batch_idx)
                train_results_list = []
                
            if ((batch_idx+1) % val_every_n_steps == 0 or self.fast_dev_run) and val_loader:
                val_results = self.validate_with(val_loader)
                val_loss =val_results['val_loss'].item()
                val_acc = val_results['val_acc'].item()

                # Logging val results  
                if logger:
                    # logger.log_metric('val_loss', batch_idx, val_loss)
                    # logger.log_metric('val_acc', batch_idx, val_acc)
                    logger['metrics/'+'val_loss'].log(val_loss, step=batch_idx)
                    logger['metrics/'+'val_acc'].log(val_acc, step=batch_idx)
                
                # Print training loss 
                step_str = f"\033[92mStep {batch_idx+1:4d}\033[0m"
                train_result_str = "\033[31m"
                for key, val in OrderedDict(train_step_result).items():
                    train_result_str += key 
                    train_result_str += ": "
                    train_result_str += f"{val.item():3.8f}"
                    train_result_str += " "
                train_result_str += "\033[0m"
                val_result_str =  f"\033[96mval_loss: {val_loss:3.8f}   val_acc: {val_acc:3.8f}\033[0m"
                print(step_str + " || " + train_result_str +" || "+ val_result_str)

            if self.fast_dev_run:
                break 

@hydra.main(version_base='1.1', config_path='conf', config_name="training_config")
def run_training(cfg: DictConfig):

    ################# Prepare data #################
    ## datamodules are from PyTorch LightningDataModule class 
    
    datamodule=None 
    if cfg.dataset.name == 'copymemory':
        datamodule = datamodules.CopyMemoryDataModule(
            delay=cfg.dataset.seq_len,
            batch_size=cfg.batch_size,
            n_train=cfg.dataset.n_train,
            n_val=cfg.dataset.n_val,
            n_test=cfg.dataset.n_test,
            n_workers=1,
            random_seeds={
                'train': random.randint(1, 10000),
                'val': random.randint(10001, 20000),
                'test': random.randint(20001, 30000) 
            }
        )

    if cfg.dataset.name == 'expandsequence':
        datamodule = datamodules.ExpandSequenceDataModule(
            seq_len=cfg.dataset.seq_len,
            vocab_size=cfg.dataset.input_dim,
            batch_size=cfg.batch_size,
            n_train=cfg.dataset.n_train,
            n_val=cfg.dataset.n_val,
            n_test=cfg.dataset.n_test,
            n_workers=1, 
            random_seeds={
                'train': random.randint(1, 10000), 
                'val': random.randint(10001, 20000), 
                'test': random.randint(20001, 30000) 
                }
        )

    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader() 
    
    # datamodule.setup(stage='test')
    # test_loader = datamodule.test_dataloader()
    
    
    ################# Define model and set up training #################

    if cfg.device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:'+str(cfg.device))

    model = SimpleRNN(
        method                  = cfg.method.name, 
        in_dim                  = cfg.dataset.input_dim,
        hidden_dim              = cfg.hidden_dim,
        out_dim                 = cfg.dataset.output_dim,
        actv_fn                 = cfg.actv_fn,
        update_g_only_before    = cfg.g_only_before_this, 
        device                  = device) 

    model.to(model.device)

    ##################### Set up optimization  ###############

    model.set_optimizers(dict(cfg.method))    


    ################# Initialize logger #################

    # # Use comet_ml logger 
    # if cfg.if_logger:
    #     experiment = Experiment(
    #         project_name='targetprop-rnn',
    #         auto_metric_step_rate=cfg.log_interval,
    #         disabled=False # for debugging 
    #     )
    #     # experiment.set_code()
    #     print(f'{hydra.utils.get_original_cwd()}')
    #     experiment.log_code(os.path.join(hydra.utils.get_original_cwd(), 'datamodules.py'))
    #     experiment.log_code(os.path.join(hydra.utils.get_original_cwd(), 'train_rnn_v2.py'))
    #     experiment.log_code(os.path.join(hydra.utils.get_original_cwd(), 'utils.py'))
    #     experiment.log_parameters(dict(cfg))
    # else:
    #     experiment=None 

    if cfg.use_neptune_logger:
        # files = ['train_rnn_v2.py', 'datamodules.py', 'utils.py']
        files_to_log = cfg.neptune_exp.upload_files
        files_to_log = [os.path.join(hydra.utils.get_original_cwd(), file) for file in files_to_log]
        tags = ['delay_'+str(cfg.dataset.seq_len)]

        run = neptune.init(project=cfg.neptune_project, source_files=files_to_log)
        run['parameters'] = dict(cfg)
        run['sys/tags'].add(tags)
        # project = neptune.init(cfg.neptune_project)
        # experiment = project.create_experiment(
        #     name=cfg.neptune_exp.name,
        #     upload_source_files=files_to_log,
        #     params=dict(cfg)
        # )
    else:
        run=None 
    

    ################# Start training loop #################

    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        logger=run,
        fast_dev_run=False,
        log_every_n_steps=cfg.log_interval,
        val_every_n_steps=cfg.val_interval
    )
     
if __name__ == "__main__":
    run_training()

