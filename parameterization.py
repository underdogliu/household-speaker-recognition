import torch


def real2pos(x):
    # return F.softplus(x)
    return torch.exp(x)


def pos2real(x):
    # return torch.where(x > torch.zeros_like(x), x + torch.log(1. - torch.exp(-x)), torch.log(torch.exp(x) - 1.))
    return torch.log(x + 1e-16)
