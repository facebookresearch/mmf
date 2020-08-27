# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import Mock

import torch
from mmf.common.meter import Meter
from mmf.common.registry import registry
from mmf.common.report import Report
from mmf.models.base_model import BaseModel
from mmf.trainers.callbacks.logistics import LogisticsCallback
from mmf.utils.file_io import PathManager
from mmf.utils.logger import setup_logger
from omegaconf import OmegaConf


class SimpleModule(BaseModel):
    def __init__(self, config={}):
        super().__init__(config)
        self.base = torch.nn.Sequential(
            torch.nn.Linear(5, 4), torch.nn.Tanh(), torch.nn.Linear(4, 5)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(5, 4), torch.nn.Tanh(), torch.nn.Linear(4, 5)
        )

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, target):
        x = self.classifier(self.base(x))
        return {"losses": {"total_loss": self.loss(x, target)}}


class NumbersDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.samples = list(range(1, 1001))

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class TestLogisticsCallback(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.trainer = argparse.Namespace()
        self.config = OmegaConf.create(
            {
                "model": "simple",
                "model_config": {},
                "training": {
                    "checkpoint_interval": 1,
                    "evaluation_interval": 10,
                    "early_stop": {"criteria": "val/total_loss"},
                    "batch_size": 16,
                    "log_interval": 10,
                    "logger_level": "info",
                },
                "env": {"save_dir": self.tmpdir},
            }
        )
        # Keep original copy for testing purposes
        self.trainer.config = deepcopy(self.config)
        registry.register("config", self.trainer.config)
        setup_logger.cache_clear()
        setup_logger()
        self.report = Mock(spec=Report)
        self.report.dataset_name = "abcd"
        self.report.dataset_type = "test"

        self.trainer.model = SimpleModule()
        self.trainer.val_dataset = NumbersDataset()

        self.trainer.optimizer = torch.optim.Adam(
            self.trainer.model.parameters(), lr=1e-01
        )
        self.trainer.device = "cpu"
        self.trainer.num_updates = 0
        self.trainer.current_iteration = 0
        self.trainer.current_epoch = 0
        self.trainer.max_updates = 0
        self.trainer.meter = Meter()
        self.cb = LogisticsCallback(self.config, self.trainer)

    def tearDown(self):
        registry.unregister("config")

    def test_on_train_start(self):
        self.cb.on_train_start()
        expected = 0
        self.assertEqual(
            int(self.cb.train_timer.get_time_since_start().split("ms")[0]), expected
        )

    def test_on_update_end(self):
        self.cb.on_train_start()
        self.cb.on_update_end(meter=self.trainer.meter, should_log=False)
        f = PathManager.open(os.path.join(self.tmpdir, "train.log"))
        self.assertFalse(any("time_since_start" in line for line in f.readlines()))
        self.cb.on_update_end(meter=self.trainer.meter, should_log=True)
        f = PathManager.open(os.path.join(self.tmpdir, "train.log"))
        self.assertTrue(any("time_since_start" in line for line in f.readlines()))

    def test_on_validation_start(self):
        self.cb.on_train_start()
        self.cb.on_validation_start()
        expected = 0
        self.assertEqual(
            int(self.cb.snapshot_timer.get_time_since_start().split("ms")[0]), expected
        )

    def test_on_test_end(self):
        self.cb.on_test_end(report=self.report, meter=self.trainer.meter)
        f = PathManager.open(os.path.join(self.tmpdir, "train.log"))
        self.assertTrue(any("Finished run in" in line for line in f.readlines()))
