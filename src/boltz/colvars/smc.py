import math
import torch
import numpy as np

def compute_ess(w):
    """
    Given a rank two tensor w whose rows are the pmf weight vectors,
    compute the effective sample size associated to each row.
    """
    wsq = torch.sum(pow(w,2),dim=1)
    ess = pow(wsq,-1)

    return ess

def icdf(w,u):
    """
    Given a rank two tensor w whose rows are pmf weight vectors and
    a rank two tensor whose rows are iid [0,1] uniform samples,
    return the rank two tensor whose rows are the 0-based indices
    of the components matching u's rows according to the cdf.
    """

    cdf = torch.cumsum(w,dim=1)
    ids = torch.searchsorted(cdf,u)

    return ids

def stratified(w,n):
    """
    Given a rank two tensor w whose rows are pmf weight vectors and
    the number of desired samples n, output a rank rwo tensor whose rows
    are random, stratified samples from their respective pmf's. The
    samples are returned as the 0-based column index corresponding to their
    component.
    """

    shift = torch.arange(n,device=w.device,dtype=w.dtype).unsqueeze(0)
    scale = torch.tensor([1/n],device=w.device,dtype=w.dtype)

    u = torch.rand(w.shape[0],n,device=w.device,dtype=w.dtype)
    u_strat = scale*(shift + u)
    ids = icdf(w,u_strat)

    return ids
