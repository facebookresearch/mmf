import argparse

import torch


def compare_tensors(a, b):
    return torch.all(a.eq(b))


def dummy_args():
    args = argparse.Namespace()
    args.opts = ["model=cnn_lstm", "dataset=clevr"]
    args.config_override = None
    return args
