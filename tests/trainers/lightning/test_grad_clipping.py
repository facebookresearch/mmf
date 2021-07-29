# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock, patch

import torch
from mmf.utils.general import clip_gradients
from pytorch_lightning.callbacks.base import Callback
from tests.trainers.test_utils import (
    get_config_with_defaults,
    get_lightning_trainer,
    get_mmf_trainer,
)


class TestLightningTrainerGradClipping(unittest.TestCase, Callback):
    def setUp(self):
        self.mmf_grads = []
        self.lightning_grads = []
        self.grad_clip_magnitude = 0.15

    def test_grad_clipping_and_parity_to_mmf(self):
        config = self._get_mmf_config(
            max_updates=5,
            max_epochs=None,
            max_grad_l2_norm=self.grad_clip_magnitude,
            clip_norm_mode="all",
        )
        mmf_trainer = get_mmf_trainer(config=config)
        mmf_trainer.evaluation_loop = MagicMock(return_value=(None, None))

        def _finish_update():
            clip_gradients(
                mmf_trainer.model,
                mmf_trainer.optimizer,
                mmf_trainer.num_updates,
                None,
                mmf_trainer.config,
            )
            for param in mmf_trainer.model.parameters():
                mmf_grad = torch.clone(param.grad).detach().item()
                self.mmf_grads.append(mmf_grad)

            mmf_trainer.scaler.step(mmf_trainer.optimizer)
            mmf_trainer.scaler.update()
            mmf_trainer.num_updates += 1

        mmf_trainer._finish_update = _finish_update
        mmf_trainer.training_loop()

        with patch("mmf.trainers.lightning_trainer.get_mmf_env", return_value=""):
            config = self._get_config(
                max_steps=5, max_epochs=None, gradient_clip_val=self.grad_clip_magnitude
            )
            trainer = get_lightning_trainer(config=config)
            trainer.callbacks.append(self)
            trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx):
        for param in pl_module.parameters():
            self.assertLessEqual(param.grad, self.grad_clip_magnitude)

        for lightning_param in pl_module.parameters():
            lightning_grad = torch.clone(lightning_param.grad).detach().item()
            self.lightning_grads.append(lightning_grad)

    def on_train_end(self, trainer, pl_module):
        for lightning_grad, mmf_grad in zip(self.lightning_grads, self.mmf_grads):
            self.assertAlmostEqual(lightning_grad, mmf_grad, places=6)

    def _get_config(self, max_steps, max_epochs, gradient_clip_val):
        config = {
            "trainer": {
                "params": {
                    "max_steps": max_steps,
                    "max_epochs": max_epochs,
                    "gradient_clip_val": gradient_clip_val,
                }
            }
        }
        return get_config_with_defaults(config)

    def _get_mmf_config(
        self, max_updates, max_epochs, max_grad_l2_norm, clip_norm_mode
    ):
        config = {
            "training": {
                "max_updates": max_updates,
                "max_epochs": max_epochs,
                "clip_gradients": True,
                "max_grad_l2_norm": max_grad_l2_norm,
                "clip_norm_mode": clip_norm_mode,
            }
        }
        return get_config_with_defaults(config)
