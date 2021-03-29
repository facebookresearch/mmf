# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock

import torch
from mmf.utils.general import clip_gradients
from pytorch_lightning.callbacks.base import Callback
from tests.trainers.lightning.test_utils import get_lightning_trainer, get_mmf_trainer


class TestLightningTrainerGradClipping(unittest.TestCase, Callback):
    def setUp(self):
        self.mmf_grads = []
        self.lightning_grads = []

        self.grad_clip_magnitude = 0.15
        self.grad_clipping_config = {
            "max_grad_l2_norm": self.grad_clip_magnitude,
            "clip_norm_mode": "all",
        }

    def test_grad_clipping_and_parity_to_mmf(self):
        mmf_trainer = get_mmf_trainer(
            max_updates=5,
            max_epochs=None,
            grad_clipping_config=self.grad_clipping_config,
        )
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

        trainer = get_lightning_trainer(
            max_steps=5,
            max_epochs=None,
            gradient_clip_val=self.grad_clip_magnitude,
            callback=self,
        )
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)

    def on_after_backward(self, trainer, pl_module):
        for param in pl_module.parameters():
            self.assertLessEqual(param.grad, self.grad_clip_magnitude)

        for lightning_param in pl_module.parameters():
            lightning_grad = torch.clone(lightning_param.grad).detach().item()
            self.lightning_grads.append(lightning_grad)

    def on_train_end(self, trainer, pl_module):
        for lightning_grad, mmf_grad in zip(self.lightning_grads, self.mmf_grads):
            self.assertAlmostEqual(lightning_grad, mmf_grad, places=6)
