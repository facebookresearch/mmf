# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import unittest
from copy import deepcopy

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.trainers.callbacks.lr_scheduler import LRSchedulerCallback
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
        self.trainer = argparse.Namespace()
        self.config = OmegaConf.create(
            {
                "model": "simple",
                "model_config": {},
                "training": {
                    "lr_scheduler": True,
                    "lr_ratio": 0.1,
                    "lr_steps": [1, 2],
                    "use_warmup": False,
                },
            }
        )
        # Keep original copy for testing purposes
        self.trainer.config = deepcopy(self.config)
        registry.register("config", self.trainer.config)

        self.trainer.model = SimpleModule()
        self.trainer.val_dataset = NumbersDataset()

        self.trainer.optimizer = torch.optim.Adam(
            self.trainer.model.parameters(), lr=1e-01
        )
        self.trainer.lr_scheduler_callback = LRSchedulerCallback(
            self.config, self.trainer
        )

    def tearDown(self):
        registry.unregister("config")

    def test_on_update_end(self):
        self.trainer.lr_scheduler_callback.on_update_end()
        self.assertAlmostEqual(self.trainer.optimizer.param_groups[0]["lr"], 1e-02)

        self.trainer.lr_scheduler_callback.on_update_end()
        self.assertAlmostEqual(self.trainer.optimizer.param_groups[0]["lr"], 1e-03)
