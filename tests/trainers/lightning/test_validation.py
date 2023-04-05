# Copyright (c) Facebook, Inc. and its affiliates.

import gc
import unittest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from mmf.common.meter import Meter
from mmf.trainers.callbacks.logistics import LogisticsCallback
from mmf.trainers.lightning_core.loop_callback import LightningLoopCallback
from mmf.trainers.lightning_core.loop_callback_with_torchmetrics import (
    LightningTorchMetricsCallback,
)
from mmf.trainers.lightning_core.torchmetric import LightningTorchMetrics
from mmf.utils.logger import TensorboardLogger
from mmf.utils.timer import Timer
from tests.trainers.test_utils import (
    get_config_with_defaults,
    get_lightning_trainer,
    get_mmf_trainer,
    run_lightning_trainer,
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
            {
                "current_iteration": 8,
                "num_updates": 8,
                "max_updates": 8,
                "avg_loss": 9701.88046875,
            },
        ]

    def teardown(self):
        del self.ground_truths
        gc.collect()

    @patch("mmf.common.test_reporter.PathManager.mkdirs")
    @patch("mmf.trainers.lightning_trainer.get_mmf_env", return_value="")
    def test_validation(self, log_dir, mkdirs):
        config = self._get_config(
            max_steps=8,
            batch_size=2,
            val_check_interval=3,
            log_every_n_steps=9,  # turn it off
            limit_val_batches=1.0,
        )
        trainer = get_lightning_trainer(config=config, prepare_trainer=False)
        callback = LightningLoopCallback(trainer)
        trainer.callbacks.append(callback)
        lightning_values = []

        def log_values(
            current_iteration: int,
            num_updates: int,
            max_updates: int,
            meter: Meter,
            extra: Dict[str, Any],
            tb_writer: TensorboardLogger,
        ):
            lightning_values.append(
                {
                    "current_iteration": current_iteration,
                    "num_updates": num_updates,
                    "max_updates": max_updates,
                    "avg_loss": meter.loss.avg,
                }
            )

        with patch(
            "mmf.trainers.lightning_core.loop_callback.summarize_report",
            side_effect=log_values,
        ):
            run_lightning_trainer(trainer)

        self.assertEqual(len(self.ground_truths), len(lightning_values))
        for gt, lv in zip(self.ground_truths, lightning_values):
            keys = list(gt.keys())
            self.assertListEqual(keys, list(lv.keys()))
            for key in keys:
                self.assertAlmostEqual(gt[key], lv[key], 1)

    # TODO: update test function with avg_loss
    @patch("mmf.common.test_reporter.PathManager.mkdirs")
    @patch("mmf.trainers.lightning_trainer.get_mmf_env", return_value="")
    def test_validation_torchmetrics(self, log_dir, mkdirs):
        config = self._get_config(
            max_steps=8,
            batch_size=2,
            val_check_interval=3,
            log_every_n_steps=9,  # turn it off
            limit_val_batches=1.0,
        )
        trainer = get_lightning_trainer(config=config, prepare_trainer=False)
        trainer.torchmetrics = LightningTorchMetrics([])
        callback = LightningTorchMetricsCallback(trainer)
        trainer.callbacks.append(callback)
        lightning_values = []

        def log_values(
            extra: Optional[Dict],
            num_updates: int,
            max_updates: int,
            log_type: str = "train",
        ):
            lightning_values.append(
                {"num_updates": num_updates, "max_updates": max_updates}
            )

        with patch(
            "mmf.trainers.lightning_core.loop_callback_with_torchmetrics"
            + ".LightningTorchMetricsCallback._log_metrics_and_extra",
            side_effect=log_values,
        ):
            run_lightning_trainer(trainer)

        self.assertEqual(len(self.ground_truths), len(lightning_values))
        for gt, lv in zip(self.ground_truths, lightning_values):
            for key in ["num_updates", "max_updates"]:
                self.assertAlmostEqual(gt[key], lv[key], 1)

    @patch("mmf.common.test_reporter.PathManager.mkdirs")
    @patch("torch.utils.tensorboard.SummaryWriter")
    @patch("mmf.common.test_reporter.get_mmf_env", return_value="")
    @patch("mmf.trainers.callbacks.logistics.summarize_report")
    def test_validation_parity(self, summarize_report_fn, test_reporter, sw, mkdirs):
        config = self._get_mmf_config(
            max_updates=8, max_epochs=None, batch_size=2, evaluation_interval=3
        )
        mmf_trainer = get_mmf_trainer(config=config)
        mmf_trainer.load_metrics()
        logistics_callback = LogisticsCallback(mmf_trainer.config, mmf_trainer)
        logistics_callback.snapshot_timer = Timer()
        logistics_callback.train_timer = Timer()
        mmf_trainer.logistics_callback = logistics_callback
        mmf_trainer.callbacks.append(logistics_callback)
        mmf_trainer.early_stop_callback = MagicMock(return_value=None)
        mmf_trainer.on_validation_end = logistics_callback.on_validation_end
        mmf_trainer.training_loop()

        calls = summarize_report_fn.call_args_list
        self.assertEqual(3, len(calls))
        self.assertEqual(len(self.ground_truths), len(calls))
        self._check_values(calls)

    def _check_values(self, calls):
        for (_, kwargs), gt in zip(calls, self.ground_truths):
            for key, value in gt.items():
                if key == "avg_loss":
                    self.assertAlmostEqual(kwargs["meter"].loss.avg, value, 1)
                else:
                    self.assertAlmostEqual(kwargs[key], value, 1)

    def _get_config(
        self,
        max_steps,
        batch_size,
        val_check_interval,
        log_every_n_steps,
        limit_val_batches,
    ):
        config = {
            "trainer": {
                "params": {
                    "max_steps": max_steps,
                    "log_every_n_steps": log_every_n_steps,
                    "val_check_interval": val_check_interval,
                    "limit_val_batches": limit_val_batches,
                }
            },
            "training": {"batch_size": batch_size},
        }
        return get_config_with_defaults(config)

    def _get_mmf_config(self, max_updates, max_epochs, batch_size, evaluation_interval):
        config = {
            "training": {
                "max_updates": max_updates,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "evaluation_interval": evaluation_interval,
            }
        }
        return get_config_with_defaults(config)
