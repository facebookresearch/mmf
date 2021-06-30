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
from mmf.utils.checkpoint import get_ckpt_path_from_folder
from mmf.utils.download import download_pretrained_model
from tests.trainers.test_utils import (
    get_config_with_defaults,
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
    def _assert_same_dict(self, mmf, lightning):
        self.assertSetEqual(set(mmf.keys()), set(lightning.keys()))
        for key in mmf.keys():
            if hasattr(mmf[key], "mean") and mmf[key].dtype == torch.float:
                self.assertAlmostEquals(
                    mmf[key].mean().item(), lightning[key].mean().item(), 2
                )
            elif hasattr(mmf[key], "item"):
                self.assertEqual(mmf[key].item(), lightning[key].item())
            else:
                self.assertEqual(mmf[key], lightning[key])

    def _get_ckpt_config(
        self, is_pl=False, ckpt_config=None, max_steps=6, resume_from_checkpoint=None
    ):
        if ckpt_config is None:
            ckpt_config = {}

        if not is_pl:
            return get_config_with_defaults(
                {
                    "training": {
                        "max_updates": max_steps,
                        "max_epochs": None,
                        "early_stop": {
                            "enabled": True,
                            "criteria": "numbers/accuracy",
                            "minimize": False,
                        },
                        "checkpoint_interval": 2,
                        "evaluation_interval": 2,
                    },
                    "model": "simple_model",
                    "evaluation": {"metrics": ["accuracy"]},
                    "checkpoint": {"max_to_keep": 1, **ckpt_config},
                    "run_type": "train_val",
                }
            )
        else:
            return get_config_with_defaults(
                {
                    "training": {
                        "checkpoint_interval": 2,
                        "early_stop": {
                            "enabled": True,
                            "criteria": "numbers/accuracy",
                            "minimize": False,
                        },
                    },
                    "trainer": {
                        "params": {
                            "max_steps": max_steps,
                            "max_epochs": None,
                            "checkpoint_callback": True,
                            "resume_from_checkpoint": resume_from_checkpoint,
                            "val_check_interval": 2,
                        }
                    },
                    "model": "simple_lightning_model",
                    "evaluation": {"metrics": ["accuracy"]},
                    "checkpoint": {"max_to_keep": 1, **ckpt_config},
                    "run_type": "train_val",
                }
            )

    def _get_mmf_trainer(
        self, ckpt_config=None, model_config=None, seed=2, max_updates=6
    ):
        config = self._get_ckpt_config(ckpt_config=ckpt_config, max_steps=max_updates)

        load_model_from_config = False
        if model_config:
            config.model_config = model_config
            config.model = list(model_config.keys())[0]
            load_model_from_config = True

        mmf_trainer = get_mmf_trainer(
            config=config, load_model_from_config=load_model_from_config, seed=seed
        )
        mmf_trainer.load_metrics()

        checkpoint_callback = CheckpointCallback(config, mmf_trainer)
        mmf_trainer.on_init_start = checkpoint_callback.on_init_start
        mmf_trainer.on_train_end = checkpoint_callback.on_train_end
        mmf_trainer.callbacks.append(checkpoint_callback)
        mmf_trainer.checkpoint_callback = checkpoint_callback

        mmf_trainer.lr_scheduler_callback = None

        early_stop_callback = EarlyStoppingCallback(config, mmf_trainer)
        mmf_trainer.early_stop_callback = early_stop_callback
        mmf_trainer.callbacks.append(early_stop_callback)

        return mmf_trainer

    def _get_lightning_trainer(
        self,
        ckpt_config=None,
        model_config=None,
        seed=2,
        max_steps=6,
        resume_from_checkpoint=None,
    ):
        config = self._get_ckpt_config(
            ckpt_config=ckpt_config,
            max_steps=max_steps,
            is_pl=True,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        load_model_from_config = False
        if model_config:
            config.model_config = model_config
            config.model = list(model_config.keys())[0]
            load_model_from_config = True

        lightning = get_lightning_trainer(
            config=config,
            prepare_trainer=False,
            load_model_from_config=load_model_from_config,
            seed=seed,
        )
        callback = LightningLoopCallback(lightning)
        lightning.callbacks.append(callback)
        lightning.callbacks += lightning.configure_checkpoint_callbacks()
        lightning.callbacks += lightning.configure_monitor_callbacks()
        prepare_lightning_trainer(lightning)
        return lightning


class TestLightningCheckpointLoadingSaving(TestLightningCheckpoint):
    def test_load_resume_parity_with_mmf(self):
        # with checkpoint.resume = True, by default it loads "current.ckpt"
        self._load_checkpoint("current.ckpt", ckpt_config={"resume": True})

    def test_load_resume_best_parity_with_mmf(self):
        # with checkpoint.resume = True and checkpoint.resume_best = True
        # by default it loads best.ckpt. It should load the "best.ckpt"
        self._load_checkpoint(
            "best.ckpt", ckpt_config={"resume": True, "resume_best": True}
        )

    def test_load_resume_ignore_resume_zoo(self):
        # specifying both checkpoint.resume = True and resume_zoo
        # resume zoo should be ignored. It should load the "current.ckpt"
        self._load_checkpoint(
            "current.ckpt",
            ckpt_config={"resume": True, "resume_zoo": "visual_bert.pretrained.coco"},
        )

    def test_load_resume_zoo_parity_with_mmf(self):
        # not specifying checkpoint.resume, but specifying
        # checkpoint.resume_zoo. It should load the file underlying zoo
        # mmf, use mmbt model
        resume_zoo = "unimodal_text.hateful_memes.bert"
        ckpt_filepath = download_pretrained_model(resume_zoo)
        ckpt_filepath = get_ckpt_path_from_folder(ckpt_filepath)
        ckpt = torch.load(ckpt_filepath, map_location="cpu")

        ckpt_config = {"resume_zoo": resume_zoo, "zoo_config_override": True}
        model_config = {
            "unimodal_text": {
                "text_hidden_size": 1,
                "classifier": {
                    "type": "mlp",
                    "params": {"num_layers": 2, "hidden_dim": 5, "out_dim": 2},
                    "losses": [{"type": "cross_entropy"}],
                },
            }
        }

        with mock_env_with_temp("mmf.utils.checkpoint.get_mmf_env") as _:
            mmf_trainer = self._get_mmf_trainer(
                ckpt_config=ckpt_config, model_config=model_config, max_updates=0
            )
            mmf_trainer.on_init_start()
            model_state_dict = mmf_trainer.model.state_dict()
            model_state_dict.pop("base.encoder.embeddings.position_ids")
            self._assert_same_dict(ckpt, model_state_dict)

        # TODO @sash: lightning side would require the mmf_checkpoint
        # to lightning adaptation. Next PR.

    def test_load_trainer_resume_zoo_parity_with_mmf(self):
        # TODO @sash: lightning side would require the mmf_checkpoint
        # to lightning adaptation. Next PR.
        pass

    def test_load_trainer_resume_file_parity_with_mmf(self):
        filename = "current.ckpt"
        mmf_ckpt_current = self._get_mmf_ckpt(filename, ckpt_config={"resume": True})

        with mock_env_with_temp("mmf.trainers.lightning_trainer.get_mmf_env") as tmp_d:
            lightning = self._get_lightning_trainer(max_steps=6)
            lightning.trainer.fit(
                lightning.model, train_dataloader=lightning.train_loader
            )

            lightning = self._get_lightning_trainer(
                max_steps=6, resume_from_checkpoint=os.path.join(tmp_d, filename)
            )
            lightning.trainer.fit(
                lightning.model, train_dataloader=lightning.train_loader
            )
            lightning_ckpt_current = torch.load(os.path.join(tmp_d, filename))
            self._assert_same_dict(
                lightning_ckpt_current["state_dict"], lightning.model.state_dict()
            )

        # Make sure lightning and mmf parity
        self._assert_same_dict(
            mmf_ckpt_current["model"], lightning_ckpt_current["state_dict"]
        )

    def test_load_trainer_ckpt_number_of_steps(self):
        with mock_env_with_temp("mmf.trainers.lightning_trainer.get_mmf_env") as tmp_d:
            # to generate ckpt file, max_steps is saved as 6
            lightning_gen = self._get_lightning_trainer(max_steps=6)
            lightning_gen.trainer.fit(
                lightning_gen.model,
                train_dataloaders=lightning_gen.train_loader,
                val_dataloaders=lightning_gen.val_loader,
            )

            # load ckpt file using resume_file, and train with max_steps 12
            resume_file = os.path.join(tmp_d, "current.ckpt")
            lightning = self._get_lightning_trainer(
                model_config={"simple_lightning_model": {"in_dim": 1}},
                resume_from_checkpoint=resume_file,
                seed=4,
                max_steps=12,
            )

            # training will take place 6 times.
            with patch.object(
                LightningLoopCallback, "on_train_batch_end", return_value=None
            ) as mock_method:
                lightning.trainer.fit(
                    lightning.model, train_dataloaders=lightning.train_loader
                )
                self.assertEquals(lightning.trainer.global_step, 12)
                call_args_list = [l[0][4] for l in mock_method.call_args_list]
                self.assertListEqual(list(range(6)), call_args_list)

    def test_save_current_parity_with_mmf(self):
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

    def _get_mmf_ckpt(self, filename, ckpt_config=None):
        with mock_env_with_temp(
            "mmf.utils.checkpoint.get_mmf_env"
        ) as tmp_d, mock_env_with_temp("mmf.common.test_reporter.get_mmf_env") as _:
            # generate checkpoint
            self._get_mmf_trainer(max_updates=6).training_loop()

            # load the generated checkpoint, calling on_init_start is
            # necessary to load the checkpoint
            mmf_trainer = self._get_mmf_trainer(
                ckpt_config=ckpt_config, max_updates=0, seed=1
            )
            mmf_trainer.on_init_start()

            mmf_ckpt_current = torch.load(os.path.join(tmp_d, filename))
            self._assert_same_dict(
                mmf_ckpt_current["model"], mmf_trainer.model.state_dict()
            )
        return mmf_ckpt_current

    def _load_checkpoint(self, filename, ckpt_config=None):
        # Make sure it loads x.ckpt when mmf
        mmf_ckpt_current = self._get_mmf_ckpt(filename, ckpt_config=ckpt_config)

        # Make sure it loads x.ckpt when lightning
        with mock_env_with_temp("mmf.trainers.lightning_trainer.get_mmf_env") as tmp_d:
            # generate checkpoint
            lightning_gen = self._get_lightning_trainer(max_steps=6)
            lightning_gen.trainer.fit(
                lightning_gen.model,
                train_dataloader=lightning_gen.train_loader,
                val_dataloaders=lightning_gen.val_loader,
            )

            # load the generated checkpoint, calling fit is necessary to load the
            # checkpoint
            lightning_ckpt_current = torch.load(os.path.join(tmp_d, filename))
            lightning = self._get_lightning_trainer(
                ckpt_config=ckpt_config, max_steps=6, seed=4
            )
            lightning.trainer.fit(
                lightning.model, train_dataloader=lightning.train_loader
            )
            self._assert_same_dict(
                lightning_ckpt_current["state_dict"], lightning.model.state_dict()
            )

        # Make sure lightning and mmf parity
        self._assert_same_dict(
            mmf_ckpt_current["model"], lightning_ckpt_current["state_dict"]
        )
