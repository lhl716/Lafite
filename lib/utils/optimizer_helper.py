import torch


def get_optimizer(parameters, config):
    return getattr(torch.optim, config.type)(parameters, **config.kwargs)

def get_scheduler(optimizer, config):
    return getattr(torch.optim.lr_scheduler, config.type)(optimizer, **config.kwargs)