# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import contextlib
import itertools
import json
import os
import platform
import random
import socket
import tempfile
import unittest
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.models.base_model import BaseModel
from mmf.utils.general import get_current_device
from omegaconf import OmegaConf
from torch import Tensor


def compare_tensors(a, b):
    return torch.equal(a, b)


def dummy_args(model="cnn_lstm", dataset="clevr", config=None):
    if config is None:
        config = os.path.join("configs", "defaults.yaml")
    args = argparse.Namespace()
    args.opts = [
        f"model={model}",
        f"dataset={dataset}",
        f"datasets={dataset}",
        f"config={config}",
    ]
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


def is_fb():
    return (
        os.getenv("SANDCASTLE") == "1"
        or os.getenv("TW_JOB_USER") == "sandcastle"
        or (
            socket.gethostname().startswith("dev")
            and not socket.gethostname().startswith("devfair")
        )
        or "fbinfra" in socket.gethostname()
    )


def skip_if_no_network(testfn, reason="Network is not available"):
    return unittest.skipUnless(NETWORK_AVAILABLE, reason)(testfn)


def skip_if_no_cuda(testfn, reason="Cuda is not available"):
    return unittest.skipUnless(CUDA_AVAILBLE, reason)(testfn)


def skip_if_windows(testfn, reason="Doesn't run on Windows"):
    return unittest.skipIf("Windows" in platform.system(), reason)(testfn)


def skip_if_macos(testfn, reason="Doesn't run on MacOS"):
    return unittest.skipIf("Darwin" in platform.system(), reason)(testfn)


def skip_if_non_fb(testfn, reason="Doesn't run on non FB infra"):
    return unittest.skipUnless(is_fb(), reason)(testfn)


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


@contextlib.contextmanager
def make_temp_dir():
    temp_dir = tempfile.TemporaryDirectory()
    try:
        yield temp_dir.name
    finally:
        # Don't clean up on Windows, as it always results in an error
        if "Windows" not in platform.system():
            temp_dir.cleanup()


def build_random_sample_list():
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
    def __init__(
        self,
        num_examples: int,
        data_item_key: str = DATA_ITEM_KEY,
        always_one: bool = False,
        dataset_type: str = "train",
    ):
        self.num_examples = num_examples
        self.data_item_key = data_item_key
        self.always_one = always_one
        self.dataset_name = "numbers"
        self.dataset_type = dataset_type

    def __getitem__(self, idx: int) -> Sample:
        sample = Sample()
        sample[self.data_item_key] = torch.tensor(idx, dtype=torch.float32).unsqueeze(
            -1
        )
        if self.always_one:
            sample.targets = torch.tensor(0, dtype=torch.long)
        return sample

    def __len__(self) -> int:
        return self.num_examples


@registry.register_model("simple_model")
class SimpleModel(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        in_dim: int = 1
        out_dim: int = 1
        data_item_key: str = DATA_ITEM_KEY

    def __init__(self, config: Config, *args, **kwargs):
        config = OmegaConf.merge(OmegaConf.structured(self.Config), config)
        super().__init__(config)
        self.data_item_key = config.data_item_key

    def build(self):
        self.classifier = torch.nn.Linear(self.config.in_dim, self.config.out_dim)

    def forward(self, prepared_batch: Dict[str, Tensor]):
        input_sample = SampleList(prepared_batch)
        batch = prepared_batch[self.data_item_key]
        output = self.classifier(batch)
        loss = torch.nn.MSELoss()(-1 * output, batch)

        return {
            "losses": {"loss": loss},
            "logits": output,
            "scores": output,
            "input_batch": input_sample,
            "dataset_type": input_sample["dataset_type"],
            "dataset_name": input_sample["dataset_name"],
        }


class SimpleNaNLossModel(SimpleModel):
    def forward(self, prepared_batch: Dict[str, Tensor]):
        report = super().forward(prepared_batch)
        report["losses"]["loss"] /= 0.0  # create an NaN loss
        return report


@registry.register_model("simple_lightning_model")
class SimpleLightningModel(SimpleModel):
    def __init__(self, config: SimpleModel.Config):
        super().__init__(config)

    def build_meters(self, run_type):
        from mmf.utils.build import build_meters

        self.train_meter, self.val_meter, self.test_meter = build_meters(run_type)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        return self._forward_step(batch, batch_idx, *args, **kwargs)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        return self._forward_step(batch, batch_idx, *args, **kwargs)

    def _forward_step(self, batch, batch_idx, *args, **kwargs):
        output = self(batch)
        output["loss"] = output["losses"]["loss"]
        return output

    def configure_optimizers(self):
        config = registry.get("config")
        if config is None:
            return torch.optim.Adam(self.parameters(), lr=0.01)
        else:
            from mmf.utils.build import build_lightning_optimizers

            return build_lightning_optimizers(self, config)


def assertModulesEqual(mod1, mod2):
    for p1, p2 in itertools.zip_longest(mod1.parameters(), mod2.parameters()):
        return p1.equal(p2)


def setup_proxy():
    # Enable proxy in FB dev env
    if not is_network_reachable() and (
        os.getenv("SANDCASTLE") == "1"
        or os.getenv("TW_JOB_USER") == "sandcastle"
        or socket.gethostname().startswith("dev")
        or "fbinfra" in socket.gethostname()
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

    model = model.to(get_current_device())
    test_sample_list = test_sample_list.to(get_current_device())

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


def search_log(log_file: str, search_condition: Optional[List[Callable]] = None):
    """Searches a log file for a particular search conditions which can be list
    of functions and returns it back

    Args:
        log_file (str): Log file in which search needs to be performed
        search_condition (List[Callable], optional): Search conditions in form of list.
            Each corresponding to a function to test a condition. Defaults to None.

    Returns:
        JSONObject: Json representation of the search line

    Throws:
        AssertionError: If no log line is found meeting the conditions
    """
    if search_condition is None:
        search_condition = {}

    lines = []

    with open(log_file) as f:
        lines = f.readlines()

    filtered_line = None
    for line in lines:
        line = line.strip()
        if "progress" not in line:
            continue
        info_index = line.find(" : ")
        line = line[info_index + 3 :]
        res = json.loads(line)

        meets_condition = True
        for condition_fn in search_condition:
            meets_condition = meets_condition and condition_fn(res)

        if meets_condition:
            filtered_line = res
            break

    assert filtered_line is not None, "No match for search condition in log file"
    return filtered_line
