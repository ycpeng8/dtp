import torch
import numpy as np


def rand_ortho(shape, irange):
    """
    Generates an orthogonal matrix. Original code from 

    Lee, D. H. and Zhang, S. and Fischer, A. and Bengio, Y., Difference 
    Target Propagation, CoRR, abs/1412.7525, 2014

    https://github.com/donghyunlee/dtp

    Parameters
    ----------
    shape  : matrix shape
    irange : range for the matrix elements
    rng    : RandomState instance, initiated with a seed

    Returns
    -------
    An orthogonal matrix of size *shape*        
    """

    # A = - irange + 2 * irange * rng.rand(*shape)
    # U, s, V = np.linalg.svd(A, full_matrices=True)
    # return np.asarray(np.dot(U, np.dot( np.eye(U.shape[1], V.shape[0]), V )),
    #                   dtype = theano.config.floatX)

    A = -irange + 2 * irange * torch.rand(shape)
    U, s, V = torch.svd(A)
    R = U @ torch.eye(U.shape[1], V.shape[1]) @ V.T
    R = R.detach()
    R.requires_grad = True
    return R


def test_rand_ortho():
    shape = torch.Size((3, 2))
    irange = np.sqrt(6. / (3. + 2.))
    device = 'cpu'
    mat = rand_ortho(shape, irange)
    print(torch.transpose(mat, 0, 1) @ mat)
    print(mat.requires_grad)
    print(mat.grad)
    print(mat.grad_fn)


def num_correct_samples(logits, Y):
    # logits: batch_size, seq_len, vocab_size
    # Y: batch_size, seq_len # long
    pred = torch.argmax(logits.transpose(0, 1), dim=2)
    pred = torch.transpose(pred, 0, 1)
    mis_match = (Y != pred).int()
    match_sum = (torch.sum(mis_match, dim=1) == 0).int().sum()
    return match_sum
