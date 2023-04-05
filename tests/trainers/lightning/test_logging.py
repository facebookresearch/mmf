# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock, patch

from mmf.trainers.callbacks.logistics import LogisticsCallback
from mmf.trainers.lightning_core.loop_callback import LightningLoopCallback
from mmf.utils.timer import Timer
from tests.test_utils import skip_if_no_network
from tests.trainers.test_utils import (
    get_config_with_defaults,
    get_lightning_trainer,
    get_mmf_trainer,
    run_lightning_trainer,
)


class TestLightningTrainerLogging(unittest.TestCase):
    def setUp(self):
        self.mmf_tensorboard_logs = []
        self.lightning_tensorboard_logs = []

    @skip_if_no_network
    @patch("mmf.common.test_reporter.PathManager.mkdirs")
    @patch("mmf.trainers.callbacks.logistics.setup_output_folder", return_value="logs")
    @patch("mmf.trainers.lightning_trainer.setup_output_folder", return_value="logs")
    @patch("mmf.utils.logger.setup_output_folder", return_value="logs")
    @patch("torch.utils.tensorboard.SummaryWriter")
    @patch("mmf.trainers.callbacks.logistics.get_mmf_env", return_value="logs")
    @patch("mmf.common.test_reporter.get_mmf_env", return_value="logs")
    @patch("mmf.trainers.lightning_trainer.get_mmf_env", return_value="logs")
    def test_tensorboard_logging_parity(
        self,
        summary_writer,
        mmf,
        lightning,
        logistics,
        logistics_logs,
        report_logs,
        trainer_logs,
        mkdirs,
    ):
        # mmf trainer
        config = self._get_mmf_config(
            max_updates=8,
            batch_size=2,
            max_epochs=None,
            log_interval=3,
            evaluation_interval=9,
            tensorboard=True,
        )
        mmf_trainer = get_mmf_trainer(config=config)

        def _add_scalars_mmf(log_dict, iteration):
            self.mmf_tensorboard_logs.append({iteration: log_dict})

        mmf_trainer.load_metrics()
        logistics_callback = LogisticsCallback(mmf_trainer.config, mmf_trainer)
        logistics_callback.snapshot_timer = MagicMock(return_value=None)
        logistics_callback.train_timer = Timer()
        logistics_callback.tb_writer.add_scalars = _add_scalars_mmf
        mmf_trainer.logistics_callback = logistics_callback
        mmf_trainer.on_validation_end = logistics_callback.on_validation_end
        mmf_trainer.callbacks = [logistics_callback]
        mmf_trainer.early_stop_callback = MagicMock(return_value=None)
        mmf_trainer.on_update_end = logistics_callback.on_update_end
        mmf_trainer.training_loop()

        # lightning_trainer
        config = self._get_config(
            max_steps=8,
            batch_size=2,
            log_every_n_steps=3,
            val_check_interval=9,
            tensorboard=True,
        )
        trainer = get_lightning_trainer(config=config, prepare_trainer=False)

        def _add_scalars_lightning(log_dict, iteration):
            self.lightning_tensorboard_logs.append({iteration: log_dict})

        def _on_fit_start_callback():
            trainer.tb_writer.add_scalars = _add_scalars_lightning

        callback = LightningLoopCallback(trainer)
        trainer.callbacks.append(callback)

        run_lightning_trainer(trainer, on_fit_start_callback=_on_fit_start_callback)
        self.assertEqual(
            len(self.mmf_tensorboard_logs), len(self.lightning_tensorboard_logs)
        )

        for mmf, lightning in zip(
            self.mmf_tensorboard_logs, self.lightning_tensorboard_logs
        ):
            self.assertDictEqual(mmf, lightning)

    def _get_config(
        self, max_steps, batch_size, log_every_n_steps, val_check_interval, tensorboard
    ):
        config = {
            "trainer": {
                "params": {
                    "max_steps": max_steps,
                    "log_every_n_steps": log_every_n_steps,
                    "val_check_interval": val_check_interval,
                }
            },
            "training": {"batch_size": batch_size, "tensorboard": tensorboard},
        }
        return get_config_with_defaults(config)

    def _get_mmf_config(
        self,
        max_updates,
        max_epochs,
        batch_size,
        log_interval,
        evaluation_interval,
        tensorboard,
    ):
        config = {
            "training": {
                "batch_size": batch_size,
                "tensorboard": tensorboard,
                "max_updates": max_updates,
                "max_epochs": max_epochs,
                "log_interval": log_interval,
                "evaluation_interval": evaluation_interval,
            }
        }
        return get_config_with_defaults(config)
