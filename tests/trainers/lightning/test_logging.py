# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock, patch

from mmf.trainers.callbacks.logistics import LogisticsCallback
from mmf.trainers.lightning_core.loop_callback import LightningLoopCallback
from mmf.utils.timer import Timer
from tests.trainers.test_utils import (
    get_lightning_trainer,
    get_mmf_trainer,
    run_lightning_trainer_with_callback,
)


class TestLightningTrainerLogging(unittest.TestCase):
    def setUp(self):
        self.mmf_tensorboard_logs = []
        self.lightning_tensorboard_logs = []

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
        mmf_trainer = get_mmf_trainer(
            max_updates=8,
            batch_size=2,
            max_epochs=None,
            log_interval=3,
            evaluation_interval=9,
            tensorboard=True,
        )

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
        trainer = get_lightning_trainer(
            max_steps=8,
            batch_size=2,
            prepare_trainer=False,
            log_every_n_steps=3,
            val_check_interval=9,
            tensorboard=True,
        )

        def _add_scalars_lightning(log_dict, iteration):
            self.lightning_tensorboard_logs.append({iteration: log_dict})

        def _on_fit_start_callback():
            trainer.tb_writer.add_scalars = _add_scalars_lightning

        callback = LightningLoopCallback(trainer)
        run_lightning_trainer_with_callback(
            trainer, callback, on_fit_start_callback=_on_fit_start_callback
        )
        self.assertEqual(
            len(self.mmf_tensorboard_logs), len(self.lightning_tensorboard_logs)
        )

        for mmf, lightning in zip(
            self.mmf_tensorboard_logs, self.lightning_tensorboard_logs
        ):
            self.assertDictEqual(mmf, lightning)
