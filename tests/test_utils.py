# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import platform
import random
import socket
import unittest

import torch


def compare_tensors(a, b):
    return torch.equal(a, b)


def dummy_args(model="cnn_lstm", dataset="clevr"):
    args = argparse.Namespace()
    args.opts = [f"model={model}", f"dataset={dataset}"]
    args.config_override = None
    return args


def is_network_reachable():
    try:
        # check if host name can be resolved
        host = socket.gethostbyname("one.one.one.one")
        # check if host is actually reachable
        s = socket.create_connection((host, 80), 2)
        s.close()
        return True
    except OSError as e:
        if e.errno == 101:
            pass
    return False


NETWORK_AVAILABLE = is_network_reachable()
CUDA_AVAILBLE = torch.cuda.is_available()


def skip_if_no_network(testfn, reason="Network is not available"):
    return unittest.skipUnless(NETWORK_AVAILABLE, reason)(testfn)


def skip_if_no_cuda(testfn, reason="Cuda is not available"):
    return unittest.skipUnless(CUDA_AVAILBLE, reason)(testfn)


def skip_if_windows(testfn, reason="Doesn't run on Windows"):
    return unittest.skipIf("Windows" in platform.system(), reason)(testfn)


def skip_if_macos(testfn, reason="Doesn't run on MacOS"):
    return unittest.skipIf("Darwin" in platform.system(), reason)(testfn)


def compare_state_dicts(a, b):
    same = True
    same = same and (list(a.keys()) == list(b.keys()))
    if not same:
        return same

    for val1, val2 in zip(a.values(), b.values()):
        if isinstance(val1, torch.Tensor):
            same = same and compare_tensors(val1, val2)
        elif not isinstance(val2, torch.Tensor):
            same = same and val1 == val2
        else:
            same = False
        if not same:
            return same

    return same


def build_random_sample_list():
    from mmf.common.sample import Sample, SampleList

    first = Sample()
    first.x = random.randint(0, 100)
    first.y = torch.rand((5, 4))
    first.z = Sample()
    first.z.x = random.randint(0, 100)
    first.z.y = torch.rand((6, 4))

    second = Sample()
    second.x = random.randint(0, 100)
    second.y = torch.rand((5, 4))
    second.z = Sample()
    second.z.x = random.randint(0, 100)
    second.z.y = torch.rand((6, 4))

    return SampleList([first, second])


DATA_ITEM_KEY = "test"


class NumbersDataset(torch.utils.data.Dataset):
    def __init__(self, num_examples):
        self.num_examples = num_examples

    def __getitem__(self, idx):
        return {DATA_ITEM_KEY: torch.tensor(idx, dtype=torch.float32)}

    def __len__(self):
        return self.num_examples


class SimpleModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = torch.nn.Linear(size, 4)

    def forward(self, prepared_batch):
        batch = prepared_batch[DATA_ITEM_KEY]
        model_output = {"losses": {"loss": torch.sum(self.linear(batch))}}
        return model_output
