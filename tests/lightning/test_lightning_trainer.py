# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from typing import Any, Dict
from unittest.mock import MagicMock

import torch
from mmf.trainers.lightning_trainer import LightningTrainer
from mmf.utils.build import build_optimizer
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.base import Callback
from tests.test_utils import NumbersDataset, SimpleLightningModel, SimpleModel
from tests.trainers.test_training_loop import TrainerTrainingLoopMock


trainer_config = OmegaConf.create(
    {
        "run_type": "train",
        "training": {
            "detect_anomaly": False,
            "evaluation_interval": 1,
            "log_interval": 2,
            "update_frequency": 1,
            "fp16": False,
        },
        "optimizer": {"type": "adam_w", "params": {"lr": 5e-5, "eps": 1e-8}},
    }
)


class LightningTrainerMock(LightningTrainer):
    def __init__(self, callback):
        self.data_module = MagicMock()
        self._benchmark = False
        self._callbacks = []
        self._distributed = False
        self._gpus = None
        self._gradient_clip_val = False
        self._num_nodes = 1
        self._deterministic = True
        self._automatic_optimization = False
        self._callbacks = [callback]
        self.config = trainer_config
        dataset = NumbersDataset(100)
        self.data_module.train_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False
        )
        self.data_module.train_loader.current_dataset = MagicMock(return_value=dataset)

    def calculate_max_updates_test(self, max_updates, max_epochs):
        self.training_config = self.config.training
        self.training_config["max_updates"] = max_updates
        self.training_config["max_epochs"] = max_epochs

        self._calculate_max_updates()
        self._load_trainer()
        return self._max_updates


class TestLightningTrainer(unittest.TestCase, Callback):
    def setUp(self):
        self.trainer = LightningTrainerMock(self)
        self.trainer.model = SimpleLightningModel(1, config=trainer_config)
        self.trainer.model.train()
        self.lightning_losses = []

        mmf_model = SimpleModel(1)
        mmf_model.train()
        mmf_optimizer = build_optimizer(mmf_model, trainer_config)
        self.mmf_losses = []
        self.mmf_trainer = TrainerTrainingLoopMock(
            100, 5, None, config=trainer_config, optimizer=mmf_optimizer
        )
        self.mmf_trainer._extract_loss = self._extract_loss_test
        self._callback_enabled = False

    def test_epoch_over_updates(self):
        trainer = self.trainer
        max_updates = trainer.calculate_max_updates_test(2, 0.04)
        self.assertEqual(max_updates, 4)

        self._check_values(0, 0)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
        self._check_values(4, 0)

    def test_fractional_epoch(self):
        trainer = self.trainer
        max_updates = trainer.calculate_max_updates_test(None, 0.04)
        self.assertEqual(max_updates, 4)

        self._check_values(0, 0)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
        self._check_values(4, 0)

    def test_updates(self):
        trainer = self.trainer
        max_updates = trainer.calculate_max_updates_test(2, None)
        self.assertEqual(max_updates, 2)

        self._check_values(0, 0)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
        self._check_values(2, 0)

    def _check_values(self, current_iteration, current_epoch):
        trainer = self.trainer.trainer
        self.assertEqual(trainer.global_step, current_iteration)
        self.assertEqual(trainer.current_epoch, current_epoch)

    def test_loss_computation(self):
        # check to see the same losses between the two trainers
        # TODO: @sash, change this to True once PL PR #4369 is merged
        self._callback_enabled = False

        # compute mmf_trainer training losses
        self.mmf_trainer.training_loop()

        # compute lightning_trainer training losses
        trainer = self.trainer
        trainer.calculate_max_updates_test(5, None)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)

    def _extract_loss_test(self, report: Dict[str, Any]) -> torch.Tensor:
        loss_dict = report.losses
        loss = sum([loss.mean() for loss in loss_dict.values()])
        self.mmf_losses.append(loss.item())
        return loss

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self._callback_enabled:
            self.lightning_losses.append(outputs["loss"].item())

    def on_train_end(self, trainer, pl_module):
        if self._callback_enabled:
            for lightning_loss, mmf_loss in zip(self.lightning_losses, self.mmf_losses):
                self.assertEqual(lightning_loss, mmf_loss)
