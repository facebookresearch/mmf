# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import itertools
import os
import platform
import random
import socket
import tempfile
import unittest

import torch
from mmf.common.sample import Sample, SampleList


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
    def __init__(self, num_examples, data_item_key=DATA_ITEM_KEY):
        self.num_examples = num_examples
        self.data_item_key = data_item_key

    def __getitem__(self, idx):
        return {
            self.data_item_key: torch.tensor(idx, dtype=torch.float32).unsqueeze(-1)
        }

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


def assertModulesEqual(mod1, mod2):
    for p1, p2 in itertools.zip_longest(mod1.parameters(), mod2.parameters()):
        return p1.equal(p2)


def setup_proxy():
    # Enable proxy in FB dev env
    if not is_network_reachable() and (
        os.getenv("SANDCASTLE") == "1"
        or os.getenv("TW_JOB_USER") == "sandcastle"
        or socket.gethostname().startswith("dev")
    ):
        os.environ["HTTPS_PROXY"] = "http://fwdproxy:8080"
        os.environ["HTTP_PROXY"] = "http://fwdproxy:8080"


def compare_torchscript_transformer_models(model, vocab_size):
    test_sample = Sample()
    test_sample.input_ids = torch.randint(low=0, high=vocab_size, size=(128,)).long()
    test_sample.input_mask = torch.ones(128).long()
    test_sample.segment_ids = torch.zeros(128).long()
    test_sample.image_feature_0 = torch.rand((1, 100, 2048)).float()
    test_sample.image = torch.rand((3, 300, 300)).float()
    test_sample_list = SampleList([test_sample])

    with torch.no_grad():
        model_output = model(test_sample_list)

    script_model = torch.jit.script(model)
    with torch.no_grad():
        script_output = script_model(test_sample_list)

    return torch.equal(model_output["scores"], script_output["scores"])


def verify_torchscript_models(model):
    model.eval()
    script_model = torch.jit.script(model)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.jit.save(script_model, tmp)
        loaded_model = torch.jit.load(tmp.name)
    return assertModulesEqual(script_model, loaded_model)
