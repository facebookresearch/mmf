import torch


def compare_tensors(a, b):
    return torch.all(a.eq(b))
