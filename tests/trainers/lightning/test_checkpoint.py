# Copyright (c) Facebook, Inc. and its affiliates.

import contextlib
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from mmf.trainers.callbacks.checkpoint import CheckpointCallback
from mmf.trainers.callbacks.early_stopping import EarlyStoppingCallback
from mmf.trainers.lightning_core.loop_callback import LightningLoopCallback
from tests.trainers.test_utils import (
    get_config,
    get_lightning_trainer,
    get_mmf_trainer,
    prepare_lightning_trainer,
)


@contextlib.contextmanager
def mock_env_with_temp(path):
    d = tempfile.TemporaryDirectory()
    patched = patch(path, return_value=d.name)
    patched.start()
    yield d.name
    d.cleanup()
    patched.stop()


class TestLightningCheckpoint(unittest.TestCase):
    def setUp(self):
        pass

    def test_write_current_ckpt_model_state_parity_with_mmf(self):
        with mock_env_with_temp(
            "mmf.utils.checkpoint.get_mmf_env"
        ) as tmp_d, mock_env_with_temp("mmf.common.test_reporter.get_mmf_env") as _:
            mmf_trainer = self._get_mmf_trainer()
            mmf_trainer.training_loop()
            mmf_ckpt_current = torch.load(os.path.join(tmp_d, "current.ckpt"))

        with mock_env_with_temp("mmf.trainers.lightning_trainer.get_mmf_env") as tmp_d:
            lightning = self._get_lightning_trainer()
            lightning.trainer.fit(
                lightning.model, train_dataloader=lightning.train_loader
            )
            lightning_ckpt_current = torch.load(os.path.join(tmp_d, "current.ckpt"))

        self._assert_same_dict(
            mmf_ckpt_current["model"], lightning_ckpt_current["state_dict"]
        )

    def test_load_current_ckpt_model_state_parity_with_mmf(self):
        with mock_env_with_temp(
            "mmf.utils.checkpoint.get_mmf_env"
        ) as tmp_d, mock_env_with_temp("mmf.common.test_reporter.get_mmf_env") as _:
            # to generate ckpt file
            self._get_mmf_trainer().training_loop()

            # load ckpt file using resume_file
            resume_file = os.path.join(tmp_d, "current.ckpt")
            mmf_trainer = self._get_mmf_trainer(
                resume_file=resume_file, model_config={"simple_model": {}}
            )
            mmf_ckpt_current = mmf_trainer.model.state_dict()
            self._assert_same_dict(mmf_ckpt_current, torch.load(resume_file)["model"])

        with mock_env_with_temp("mmf.trainers.lightning_trainer.get_mmf_env") as tmp_d:
            # to generate ckpt file
            lightning_gen = self._get_lightning_trainer()
            lightning_gen.trainer.fit(
                lightning_gen.model, train_dataloader=lightning_gen.train_loader
            )

            # load ckpt file using resume_file
            resume_file = os.path.join(tmp_d, "current.ckpt")
            lightning = self._get_lightning_trainer(
                resume_file=resume_file, model_config={"simple_lightning_model": {}}
            )
            lightning_ckpt_current = lightning.model.state_dict()
            self._assert_same_dict(
                lightning_ckpt_current, torch.load(resume_file)["state_dict"]
            )

        self._assert_same_dict(mmf_ckpt_current, lightning_ckpt_current)
        # lightning

    def _get_mmf_trainer(self, resume_file=None, model_config=None):
        config = get_config(
            {
                "training": {
                    "max_updates": 6,
                    "max_epochs": None,
                    "early_stop": {"criteria": "numbers/accuracy", "minimize": False},
                    "checkpoint_interval": 6,
                    "evaluation_interval": 6,
                },
                "model": "simple_model",
                "evaluation": {"metrics": ["accuracy"]},
                "checkpoint": {"resume": False, "max_to_keep": 1},
                "run_type": "train",
            }
        )

        if resume_file:
            config.checkpoint.resume = True
            config.checkpoint.resume_file = resume_file

        load_model_from_config = False
        if model_config:
            config.model_config = model_config
            load_model_from_config = True

        mmf_trainer = get_mmf_trainer(
            config=config, load_model_from_config=load_model_from_config
        )
        mmf_trainer.load_metrics()

        checkpoint_callback = CheckpointCallback(config, mmf_trainer)
        mmf_trainer.callbacks.append(checkpoint_callback)
        mmf_trainer.checkpoint_callback = checkpoint_callback

        mmf_trainer.lr_scheduler_callback = None

        early_stop_callback = EarlyStoppingCallback(config, mmf_trainer)
        mmf_trainer.early_stop_callback = early_stop_callback
        mmf_trainer.callbacks.append(early_stop_callback)

        return mmf_trainer

    def _get_lightning_trainer(self, resume_file=None, model_config=None):
        config = get_config(
            {
                "training": {"checkpoint_interval": 6, "evaluation_interval": 6},
                "trainer": {
                    "params": {
                        "max_steps": 6,
                        "max_epochs": None,
                        "checkpoint_callback": True,
                    }
                },
                "model": "simple_lightning_model",
                "checkpoint": {"resume": False, "max_to_keep": 1},
            }
        )

        if resume_file:
            config.checkpoint.resume = True
            config.checkpoint.resume_file = resume_file

        load_model_from_config = False
        if model_config:
            config.model_config = model_config
            load_model_from_config = True

        lightning = get_lightning_trainer(
            config=config,
            prepare_trainer=False,
            load_model_from_config=load_model_from_config,
        )
        callback = LightningLoopCallback(lightning)
        lightning._callbacks.append(callback)
        lightning._callbacks += lightning.configure_checkpoint_callbacks()
        prepare_lightning_trainer(lightning)
        return lightning

    def _assert_same_dict(self, mmf, lightning):
        self.assertSetEqual(set(mmf.keys()), set(lightning.keys()))
        for key in mmf.keys():
            self.assertAlmostEquals(
                mmf[key].mean().item(), lightning[key].mean().item(), 2
            )
