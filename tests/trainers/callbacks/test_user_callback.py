# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
import unittest
from copy import deepcopy

import torch
from mmf.common.registry import registry
from mmf.trainers.callbacks.lr_scheduler import LRSchedulerCallback
from mmf.utils.configuration import load_yaml
from omegaconf import OmegaConf
from tests.test_utils import NumbersDataset, SimpleModel


registry.register_callback("test_callback")(LRSchedulerCallback)


class TestUserCallback(unittest.TestCase):
    def setUp(self):
        self.trainer = argparse.Namespace()
        self.config = load_yaml(os.path.join("configs", "defaults.yaml"))
        self.config = OmegaConf.merge(
            self.config,
            {
                "model": "simple",
                "model_config": {},
                "training": {
                    "lr_scheduler": True,
                    "lr_ratio": 0.1,
                    "lr_steps": [1, 2],
                    "use_warmup": False,
                    "callbacks": [{"type": "test_callback", "params": {}}],
                },
            },
        )
        # Keep original copy for testing purposes
        self.trainer.config = deepcopy(self.config)
        registry.register("config", self.trainer.config)

        model = SimpleModel(SimpleModel.Config())
        model.build()
        self.trainer.model = model
        self.trainer.val_loader = torch.utils.data.DataLoader(
            NumbersDataset(2), batch_size=self.config.training.batch_size
        )

        self.trainer.optimizer = torch.optim.Adam(
            self.trainer.model.parameters(), lr=1e-01
        )
        self.trainer.lr_scheduler_callback = LRSchedulerCallback(
            self.config, self.trainer
        )

        self.trainer.callbacks = []
        for callback in self.config.training.get("callbacks", []):
            callback_type = callback.type
            callback_param = callback.params
            callback_cls = registry.get_callback_class(callback_type)
            self.trainer.callbacks.append(
                callback_cls(self.trainer.config, self.trainer, **callback_param)
            )

    def tearDown(self):
        registry.unregister("config")

    def test_on_update_end(self):
        self.assertEqual(len(self.trainer.callbacks), 1)
        user_callback = self.trainer.callbacks[0]
        self.assertTrue(isinstance(user_callback, LRSchedulerCallback))
