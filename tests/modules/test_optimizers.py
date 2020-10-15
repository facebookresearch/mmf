# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.models.mmbt import MMBT
from mmf.utils.build import build_optimizer
from omegaconf import OmegaConf
from tests.test_utils import SimpleModel, skip_if_no_network


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {"optimizer": {"type": "adam_w", "params": {"lr": 5e-5}}}
        )

    def test_build_optimizer_simple_model(self):
        model = SimpleModel(1)

        optimizer = build_optimizer(model, self.config)
        self.assertTrue(isinstance(optimizer, torch.optim.Optimizer))
        self.assertEqual(len(optimizer.param_groups), 1)

    @skip_if_no_network
    def test_build_optimizer_custom_model(self):
        model = MMBT.from_params()
        model.build()
        self.config.model = model.config.model
        self.config.model_config = model.config

        optimizer = build_optimizer(model, self.config)
        self.assertTrue(isinstance(optimizer, torch.optim.Optimizer))
        self.assertEqual(len(optimizer.param_groups), 2)
