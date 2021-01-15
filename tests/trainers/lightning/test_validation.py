# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock, patch

from mmf.trainers.callbacks.logistics import LogisticsCallback
from mmf.trainers.lightning_core.loop_callback import LightningLoopCallback
from tests.trainers.lightning.test_utils import (
    get_lightning_trainer,
    get_mmf_trainer,
    run_lightning_trainer_with_callback,
)


class TestLightningTrainerValidation(unittest.TestCase):
    def setUp(self):
        self.ground_truths = [
            {
                "current_iteration": 3,
                "num_updates": 3,
                "max_updates": 8,
                "avg_loss": 9705.2953125,
            },
            {
                "current_iteration": 6,
                "num_updates": 6,
                "max_updates": 8,
                "avg_loss": 9703.29765625,
            },
            # TODO: @sash, add one last validation when training is done
        ]

    @patch("mmf.trainers.lightning_core.loop_callback.summarize_report")
    def test_validation(self, summarize_report_fn):
        trainer = get_lightning_trainer(
            max_steps=8, batch_size=2, prepare_trainer=False, val_check_interval=3
        )
        callback = LightningLoopCallback(trainer)
        run_lightning_trainer_with_callback(trainer, callback)
        calls = summarize_report_fn.call_args_list

        self.assertEqual(len(self.ground_truths), len(calls))
        for (args, kwargs), gt in zip(calls, self.ground_truths):
            for key, value in gt.items():
                if key == "avg_loss":
                    self.assertEqual(kwargs["meter"].loss.avg, value)
                else:
                    self.assertEqual(kwargs[key], value)

    @patch("mmf.trainers.callbacks.logistics.summarize_report")
    def test_validation_parity(self, summarize_report_fn):
        mmf_trainer = get_mmf_trainer(
            max_updates=8,
            batch_size=2,
            max_epochs=None,
            evaluation_interval=3,
            mock_functions=False,
        )
        mmf_trainer.load_metrics()
        logistics_callback = LogisticsCallback(mmf_trainer.config, mmf_trainer)
        logistics_callback.snapshot_timer = MagicMock(return_value=None)
        logistics_callback.train_timer = MagicMock(return_value=None)
        mmf_trainer.logistics_callback = logistics_callback
        mmf_trainer.callbacks.append(logistics_callback)
        mmf_trainer.early_stop_callback = MagicMock(return_value=None)
        mmf_trainer.training_loop()

        calls = summarize_report_fn.call_args_list
        self.assertEqual(3, len(calls))
        calls = calls[:2]

        self.assertEqual(len(self.ground_truths), len(calls))
        for (args, kwargs), gt in zip(calls, self.ground_truths):
            for key, value in gt.items():
                if key == "avg_loss":
                    self.assertEqual(kwargs["meter"].loss.avg, value)
                else:
                    self.assertEqual(kwargs[key], value)
