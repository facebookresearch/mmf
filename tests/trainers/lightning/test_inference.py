# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock, patch

from mmf.trainers.callbacks.checkpoint import CheckpointCallback
from mmf.trainers.callbacks.logistics import LogisticsCallback
from mmf.trainers.lightning_core.loop_callback import LightningLoopCallback
from mmf.utils.timer import Timer
from tests.trainers.test_utils import (
    get_config_with_defaults,
    get_lightning_trainer,
    get_mmf_trainer,
    mock_env_with_temp,
    run_lightning_trainer,
)


unimodal_text_model_config = {
    "unimodal_text": {
        "text_hidden_size": 1,
        "classifier": {
            "type": "mlp",
            "params": {"num_layers": 2, "hidden_dim": 5, "out_dim": 2},
            "losses": [{"type": "cross_entropy"}],
        },
    }
}


class TestLightningInference(unittest.TestCase):
    @patch("mmf.trainers.callbacks.logistics.summarize_report")
    @patch("mmf.trainers.lightning_trainer.summarize_report")
    def test_final_val_inference_parity_with_mmf(self, lightning, mmf):
        # lightning
        with mock_env_with_temp("mmf.trainers.lightning_trainer.get_mmf_env") as _:
            config = self._get_config(
                max_steps=8,
                batch_size=2,
                val_check_interval=3,
                log_every_n_steps=9,  # turn it off
                limit_val_batches=1.0,
            )
            model_config = {"simple_lightning_model": {"in_dim": 1}}
            config.model_config = model_config
            config.model = "simple_lightning_model"
            trainer = get_lightning_trainer(
                config=config, load_model_from_config=True, prepare_trainer=False
            )
            callback = LightningLoopCallback(trainer)
            trainer.callbacks.append(callback)
            run_lightning_trainer(trainer)
            trainer.try_loading_best_model()
            trainer.inference()

        # mmf
        with mock_env_with_temp(
            "mmf.utils.checkpoint.get_mmf_env"
        ) as _, mock_env_with_temp("mmf.common.test_reporter.get_mmf_env") as _:
            config = self._get_mmf_config(
                max_updates=8, max_epochs=None, batch_size=2, evaluation_interval=3
            )
            model_config = {"simple_model": {"in_dim": 1}}
            config.model_config = model_config
            config.model = "simple_model"
            mmf_trainer = get_mmf_trainer(config=config)
            mmf_trainer.load_metrics()
            logistics_callback = LogisticsCallback(mmf_trainer.config, mmf_trainer)
            logistics_callback.snapshot_timer = Timer()
            logistics_callback.train_timer = Timer()
            mmf_trainer.logistics_callback = logistics_callback
            mmf_trainer.callbacks.append(logistics_callback)

            checkpoint_callback = CheckpointCallback(config, mmf_trainer)
            mmf_trainer.on_init_start = checkpoint_callback.on_init_start
            mmf_trainer.on_train_end = checkpoint_callback.on_train_end
            mmf_trainer.callbacks.append(checkpoint_callback)
            mmf_trainer.checkpoint_callback = checkpoint_callback

            mmf_trainer.early_stop_callback = MagicMock(return_value=None)
            mmf_trainer.on_validation_end = logistics_callback.on_validation_end
            mmf_trainer.train()

        lightning_loss = lightning.call_args_list[-1][1]["meter"].meters["loss"].avg
        mmf_loss = mmf.call_args_list[-1][1]["meter"].meters["loss"].avg
        self.assertEquals(lightning_loss, mmf_loss)

    def _get_config(
        self,
        max_steps,
        batch_size,
        val_check_interval,
        log_every_n_steps,
        limit_val_batches,
        checkpoint_interval=2,
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
            "training": {
                "batch_size": batch_size,
                "checkpoint_interval": checkpoint_interval,
            },
        }
        return get_config_with_defaults(config)

    def _get_mmf_config(
        self,
        max_updates,
        max_epochs,
        batch_size,
        evaluation_interval,
        checkpoint_interval=2,
    ):
        config = {
            "training": {
                "max_updates": max_updates,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "evaluation_interval": evaluation_interval,
                "checkpoint_interval": checkpoint_interval,
            }
        }
        return get_config_with_defaults(config)
