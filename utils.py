import torch


def tensor_check(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        x = x.item() if x.dim() == 0 else x.numpy()
    return x
