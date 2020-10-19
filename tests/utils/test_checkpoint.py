# Copyright (c) Facebook, Inc. and its affiliates.

import contextlib
import os
import tempfile
import unittest
from copy import deepcopy
from io import StringIO
from unittest.mock import Mock, patch

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.trainers.callbacks.checkpoint import CheckpointCallback
from mmf.trainers.callbacks.early_stopping import EarlyStoppingCallback
from mmf.trainers.callbacks.lr_scheduler import LRSchedulerCallback
from mmf.utils.checkpoint import Checkpoint
from mmf.utils.configuration import load_yaml
from mmf.utils.file_io import PathManager
from omegaconf import OmegaConf
from tests.test_utils import compare_state_dicts, skip_if_no_cuda


@contextlib.contextmanager
def mock_env_with_temp():
    d = tempfile.TemporaryDirectory()
    patched = patch("mmf.utils.checkpoint.get_mmf_env", return_value=d.name)
    patched.start()
    yield d.name
    d.cleanup()
    patched.stop()


class SimpleModule(BaseModel):
    def __init__(self, config={}):
        super().__init__(config)
        self.base = torch.nn.Sequential(
            torch.nn.Linear(5, 4), torch.nn.Tanh(), torch.nn.Linear(4, 5)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(5, 4), torch.nn.Tanh(), torch.nn.Linear(4, 5)
        )

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, target):
        x = self.classifier(self.base(x))
        return {"losses": {"total_loss": self.loss(x, target)}}


class OnlyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_test = torch.nn.Sequential(
            torch.nn.Linear(5, 4), torch.nn.Tanh(), torch.nn.Linear(4, 5)
        )

    def format_state_key(self, key):
        return key


class TestUtilsCheckpoint(unittest.TestCase):
    def setUp(self):
        import argparse

        torch.manual_seed(1234)
        # An easy way to get a AttributeDict object
        self.trainer = argparse.Namespace()
        self.config = OmegaConf.create(
            {
                "model": "simple",
                "model_config": {},
                "checkpoint": {
                    "save_git_details": False,
                    "reset": {
                        "optimizer": False,
                        "counts": False,
                        "all": False,
                        "fp16_scaler": False,
                    },
                    "pretrained_state_mapping": {"base_test": "base"},
                    "max_to_keep": 5,
                },
                "config_override": "test",
                "training": {
                    "checkpoint_interval": 1,
                    "early_stop": {"criteria": "val/total_loss"},
                    "lr_scheduler": True,
                },
                "scheduler": {
                    "type": "multi_step",
                    "params": {
                        "use_warmup": False,
                        "lr_steps": [10, 20],
                        "lr_ratio": 0.1,
                        "warmup_factor": 1.0,
                    },
                },
            }
        )
        # Keep original copy for testing purposes
        self.trainer.config = deepcopy(self.config)

        self.trainer.model = SimpleModule()
        self.trainer.scaler = torch.cuda.amp.GradScaler()

        self.trainer.optimizer = torch.optim.Adam(
            self.trainer.model.parameters(), lr=1e-01
        )

        self.trainer.lr_scheduler_callback = LRSchedulerCallback(
            self.config, self.trainer
        )

    def test_save_config(self):
        with mock_env_with_temp() as d:
            Checkpoint(self.trainer)
            config = load_yaml(os.path.join(d, "config.yaml"))
            self.assertTrue(config == self.config)
            self.assertTrue(config == self.trainer.config)

    def test_save_and_load_state_dict(self):
        with mock_env_with_temp() as d:
            checkpoint = Checkpoint(self.trainer)
            self._init_early_stopping(checkpoint)
            self._do_a_pass()
            # Test normal case
            checkpoint.save(1500)

            self.assertTrue(
                PathManager.exists(os.path.join(d, "models", "model_1500.ckpt"))
            )
            self.assertTrue(PathManager.exists(os.path.join(d, "current.ckpt")))
            self.assertFalse(PathManager.exists(os.path.join(d, "best.ckpt")))
            os.remove(os.path.join(d, "models", "model_1500.ckpt"))
            os.remove(os.path.join(d, "current.ckpt"))

            best_model = deepcopy(self.trainer.model)
            best_optimizer = deepcopy(self.trainer.optimizer)
            # Test with update_best
            checkpoint.save(2000, update_best=True)

            self.assertTrue(
                PathManager.exists(os.path.join(d, "models", "model_2000.ckpt"))
            )
            self.assertTrue(PathManager.exists(os.path.join(d, "best.ckpt")))
            self.assertTrue(PathManager.exists(os.path.join(d, "current.ckpt")))

            self._do_a_pass()
            checkpoint.save(2500)

            # Test resume
            self.trainer.config.checkpoint.resume = True

            current_model = deepcopy(self.trainer.model)
            current_optimizer = deepcopy(self.trainer.optimizer)
            checkpoint.load_state_dict()

            self.assertFalse(
                compare_state_dicts(
                    self.trainer.model.state_dict(), best_model.state_dict()
                )
            )
            self.assertTrue(
                compare_state_dicts(
                    self.trainer.model.state_dict(), current_model.state_dict()
                )
            )
            self.assertFalse(
                self._compare_optimizers(self.trainer.optimizer, best_optimizer)
            )
            self.assertTrue(
                self._compare_optimizers(self.trainer.optimizer, current_optimizer)
            )

            base_0_weight_current = self.trainer.model.base[0].weight.data.clone()

            # Test resume_best
            self.trainer.config.checkpoint.resume = True
            self.trainer.config.checkpoint.resume_best = True

            checkpoint.load_state_dict()

            self.assertTrue(
                compare_state_dicts(
                    self.trainer.model.state_dict(), best_model.state_dict()
                )
            )
            self.assertTrue(
                self._compare_optimizers(self.trainer.optimizer, best_optimizer)
            )
            self.assertFalse(
                self._compare_optimizers(self.trainer.optimizer, current_optimizer)
            )
            base_0_weight_best = self.trainer.model.base[0].weight.data.clone()

            self.trainer.config.checkpoint.resume_best = False
            # Test distributed settings
            self.trainer.model = torch.nn.DataParallel(self.trainer.model)
            checkpoint.load_state_dict()

            weight_to_be_tested = self.trainer.model.module.base[0].weight
            weight_device = weight_to_be_tested.device

            self.assertTrue(
                torch.equal(
                    weight_to_be_tested, base_0_weight_current.to(weight_device)
                )
            )
            self.assertFalse(
                torch.equal(weight_to_be_tested, base_0_weight_best.to(weight_device))
            )

    def test_finalize_and_restore_from_it(self):
        with mock_env_with_temp():
            checkpoint = Checkpoint(self.trainer)
            self._init_early_stopping(checkpoint)
            original_model = deepcopy(self.trainer.model)
            self._do_a_pass()
            model_1500 = deepcopy(self.trainer.model)
            checkpoint.save(1500)

            swap = self.trainer.model
            self.trainer.model = original_model
            checkpoint.restore()
            # First test without best.ckpt
            self.assertTrue(
                compare_state_dicts(
                    self.trainer.model.state_dict(), original_model.state_dict()
                )
            )
            self.assertFalse(
                compare_state_dicts(
                    self.trainer.model.state_dict(), model_1500.state_dict()
                )
            )

            self.trainer.model = swap

            self._do_a_pass()
            model_2000 = deepcopy(self.trainer.model)
            checkpoint.save(2000, update_best=True)

            self._do_a_pass()
            model_2500 = deepcopy(self.trainer.model)
            checkpoint.save(2500)

            checkpoint.restore()

            self.assertFalse(
                compare_state_dicts(
                    self.trainer.model.state_dict(), original_model.state_dict()
                )
            )
            self.assertFalse(
                compare_state_dicts(
                    self.trainer.model.state_dict(), model_1500.state_dict()
                )
            )
            self.assertTrue(
                compare_state_dicts(
                    self.trainer.model.state_dict(), model_2000.state_dict()
                )
            )
            self.assertFalse(
                compare_state_dicts(
                    self.trainer.model.state_dict(), model_2500.state_dict()
                )
            )

    def test_finalize_and_resume_file(self):
        with mock_env_with_temp() as d:
            checkpoint = Checkpoint(self.trainer)
            self._init_early_stopping(checkpoint)
            self._do_a_pass()
            checkpoint.finalize()
            original = deepcopy(self.trainer.model)
            pth_path = os.path.join(d, "simple_final.pth")
            self.assertTrue(PathManager.exists(pth_path))

            self._do_a_pass()

            after_a_pass = deepcopy(self.trainer.model)
            original_optimizer = deepcopy(self.trainer.optimizer)
            self.trainer.config.checkpoint.resume_file = pth_path

            with contextlib.redirect_stdout(StringIO()):
                checkpoint.load_state_dict()
            self.assertTrue(
                compare_state_dicts(
                    self.trainer.model.state_dict(), original.state_dict()
                )
            )
            self.assertFalse(
                compare_state_dicts(
                    self.trainer.model.state_dict(), after_a_pass.state_dict()
                )
            )
            self.assertTrue(
                self._compare_optimizers(self.trainer.optimizer, original_optimizer)
            )

    def test_resets(self):
        with mock_env_with_temp():
            checkpoint = Checkpoint(self.trainer)
            self._init_early_stopping(checkpoint)
            self._do_a_pass()

            original_optimizer = deepcopy(self.trainer.optimizer)
            original_model = deepcopy(self.trainer.model)
            original_scaler = deepcopy(self.trainer.scaler)

            self.trainer.current_epoch = 3
            checkpoint.save(2000, update_best=True)
            self.trainer.current_epoch = 4
            # Test reset all
            self.trainer.config.checkpoint.resume = True
            self.trainer.config.checkpoint.reset.all = True
            checkpoint.load_state_dict()

            self.assertTrue(
                compare_state_dicts(
                    self.trainer.model.state_dict(), original_model.state_dict()
                )
            )

            self.assertTrue(
                self._compare_optimizers(self.trainer.optimizer, original_optimizer)
            )

            self.assertTrue(
                compare_state_dicts(
                    self.trainer.scaler.state_dict(), original_scaler.state_dict()
                )
            )

            self.assertEqual(self.trainer.num_updates, 0)
            self.assertEqual(self.trainer.current_iteration, 0)
            self.assertEqual(self.trainer.current_epoch, 4)

            # Test reset_optimizer
            self._init_early_stopping(checkpoint)
            self.trainer.config.checkpoint.reset.all = False
            self.trainer.config.checkpoint.reset.optimizer = True
            checkpoint.load_state_dict()

            self.assertTrue(
                compare_state_dicts(
                    self.trainer.model.state_dict(), original_model.state_dict()
                )
            )

            self.assertTrue(
                self._compare_optimizers(self.trainer.optimizer, original_optimizer)
            )

            self.assertEqual(self.trainer.num_updates, 2000)
            self.assertEqual(self.trainer.current_iteration, 2000)
            self.assertEqual(self.trainer.current_epoch, 3)

            self._init_early_stopping(checkpoint)
            # Test reset_counts
            self.trainer.config.checkpoint.reset.all = False
            self.trainer.config.checkpoint.reset.optimizer = False
            self.trainer.config.checkpoint.reset.counts = True
            checkpoint.load_state_dict()

            self.assertTrue(
                compare_state_dicts(
                    self.trainer.model.state_dict(), original_model.state_dict()
                )
            )

            self.assertTrue(
                self._compare_optimizers(self.trainer.optimizer, original_optimizer)
            )
            self.assertEqual(self.trainer.num_updates, 0)
            self.assertEqual(self.trainer.current_iteration, 0)
            self.assertEqual(self.trainer.current_epoch, 2)

            # Test with resume_best
            self._do_a_pass()
            checkpoint.save(3000)
            self._init_early_stopping(checkpoint)
            self.trainer.config.checkpoint.reset.all = False
            self.trainer.config.checkpoint.resume_best = True
            self.trainer.config.checkpoint.reset.optimizer = True
            self.trainer.config.checkpoint.reset.counts = False
            checkpoint.load_state_dict()

            self.assertTrue(
                compare_state_dicts(
                    self.trainer.model.state_dict(), original_model.state_dict()
                )
            )

            self.assertFalse(
                self._compare_optimizers(self.trainer.optimizer, original_optimizer)
            )

            self.assertEqual(self.trainer.num_updates, 1000)
            self.assertEqual(self.trainer.current_iteration, 1000)
            self.assertEqual(self.trainer.current_epoch, 3)

    @skip_if_no_cuda
    def test_checkpoint_scaler_loading(self):
        with mock_env_with_temp():
            original_scaler = deepcopy(self.trainer.scaler)

            checkpoint = Checkpoint(self.trainer)
            self._init_early_stopping(checkpoint)

            self._do_a_fp16_pass()
            checkpoint.save(1000)
            self.trainer.config.checkpoint.resume = True
            self.trainer.config.checkpoint.reset.all = False
            self.trainer.config.checkpoint.reset.optimizer = True
            self.trainer.config.checkpoint.reset.counts = True
            self.trainer.config.checkpoint.reset.fp16_scaler = True

            # Reset to make it same as the default grad scaler
            self.trainer.scaler = torch.cuda.amp.GradScaler()
            checkpoint.load_state_dict()
            self.assertTrue(
                compare_state_dicts(
                    self.trainer.scaler.state_dict(), original_scaler.state_dict()
                )
            )

            self._do_a_fp16_pass()
            checkpoint.save(2000)
            self.trainer.config.checkpoint.reset.all = False
            self.trainer.config.checkpoint.reset.optimizer = True
            self.trainer.config.checkpoint.reset.counts = True
            self.trainer.config.checkpoint.reset.fp16_scaler = False

            # Reset again to make it same as the default grad scaler
            self.trainer.scaler = torch.cuda.amp.GradScaler()
            checkpoint.load_state_dict()
            self.assertFalse(
                compare_state_dicts(
                    self.trainer.scaler.state_dict(), original_scaler.state_dict()
                )
            )

    def test_max_to_keep(self):
        with mock_env_with_temp():
            checkpoint = Checkpoint(self.trainer)
            self._init_early_stopping(checkpoint)

            ckpt_paths = []
            for indx in [2000, 3000, 4000, 5000, 6000]:
                self._do_a_pass()
                checkpoint.save(indx, update_best=False)

                ckpt_paths.append(
                    os.path.join(checkpoint.models_foldername, "model_%d.ckpt" % indx)
                )
                self.assertTrue(os.path.exists(ckpt_paths[-1]))

            for indx, u in enumerate([7000, 8000, 9000, 10000, 11000]):
                self._do_a_pass()
                checkpoint.save(u, update_best=False)

                ckpt_paths.append(
                    os.path.join(checkpoint.models_foldername, "model_%d.ckpt" % u)
                )
                self.assertTrue(os.path.exists(ckpt_paths[-1]))
                self.assertFalse(os.path.exists(ckpt_paths[indx]))

    def test_zoo_load(self):
        with mock_env_with_temp():
            checkpoint = Checkpoint(self.trainer)
            self._init_early_stopping(checkpoint)
            self._do_a_pass()

            original_model = deepcopy(self.trainer.model)
            ret_load_pretrained_zoo = {
                "config": self.config.model_config,
                "checkpoint": deepcopy(self.trainer.model.state_dict()),
                "full_config": self.config,
            }

            self._do_a_pass()

            with patch(
                "mmf.utils.checkpoint.load_pretrained_model",
                return_value=ret_load_pretrained_zoo,
            ):
                self.trainer.config.checkpoint.resume_zoo = "random"
                with contextlib.redirect_stdout(StringIO()):
                    checkpoint.load_state_dict()
                self.assertTrue(
                    compare_state_dicts(
                        self.trainer.model.state_dict(), original_model.state_dict()
                    )
                )

                # Now, test zoo override
                self.trainer.config.checkpoint.zoo_config_override = True
                SimpleModule.from_pretrained = Mock(
                    return_value=deepcopy(original_model)
                )
                registry.register_model("simple")(SimpleModule)
                with contextlib.redirect_stdout(StringIO()):
                    checkpoint.load_state_dict()
                self.assertTrue(
                    compare_state_dicts(
                        self.trainer.model.state_dict(), original_model.state_dict()
                    )
                )

    def test_pretrained_load(self):
        with mock_env_with_temp() as d:
            checkpoint = Checkpoint(self.trainer)
            self._init_early_stopping(checkpoint)
            self._do_a_pass()
            original_model = deepcopy(self.trainer.model)
            # Test with zoo now
            ret_load_pretrained_zoo = {
                "config": self.config.model_config,
                "checkpoint": deepcopy(self.trainer.model.state_dict()),
                "full_config": self.config,
            }

            checkpoint.save(2000)
            self.trainer.config.checkpoint.resume_file = os.path.join(d, "current.ckpt")
            self.trainer.config.checkpoint.resume_pretrained = True
            self.trainer.model = OnlyBase()
            checkpoint.load_state_dict()

            self.assertTrue(
                compare_state_dicts(
                    self.trainer.model.base_test.state_dict(),
                    original_model.base.state_dict(),
                )
            )

            with patch(
                "mmf.utils.checkpoint.load_pretrained_model",
                return_value=ret_load_pretrained_zoo,
            ):
                self.trainer.config.checkpoint.resume_zoo = "random"
                self.trainer.config.checkpoint.resume_file = None
                self.trainer.model = OnlyBase()
                checkpoint.load_state_dict()

                self.assertTrue(
                    compare_state_dicts(
                        self.trainer.model.base_test.state_dict(),
                        original_model.base.state_dict(),
                    )
                )

    def _init_early_stopping(self, checkpoint):
        self.trainer.num_updates = 0
        self.trainer.current_iteration = 0
        self.trainer.current_epoch = 0
        self.trainer.checkpoint_callback = CheckpointCallback(self.config, self.trainer)
        self.trainer.early_stop_callback = EarlyStoppingCallback(
            self.config, self.trainer
        )
        self.trainer.early_stop_callback.early_stopping.best_monitored_iteration = 1000
        self.trainer.early_stop_callback.early_stopping.best_monitored_update = 1000
        self.trainer.early_stop_callback.early_stopping.best_monitored_value = 0.1
        self.trainer.current_epoch = 2

    def _do_a_pass(self):
        self.trainer.optimizer.zero_grad()
        self.trainer.model.train()
        with contextlib.redirect_stdout(StringIO()):
            loss = self.trainer.model(
                torch.rand(5, 5, requires_grad=True),
                torch.empty(5, dtype=torch.long).random_(5),
            )

        loss["losses"]["total_loss"].sum().backward()
        self.trainer.optimizer.step()
        self.trainer.lr_scheduler_callback._scheduler.step()

    def _do_a_fp16_pass(self):
        self.trainer.optimizer.zero_grad()
        self.trainer.model.train()
        self.trainer.model.cuda()
        with contextlib.redirect_stdout(StringIO()):
            with torch.cuda.amp.autocast():
                loss = self.trainer.model(
                    torch.rand(5, 5, requires_grad=True).cuda(),
                    torch.empty(5, dtype=torch.long).random_(5).cuda(),
                )

        self.trainer.scaler.scale(loss["losses"]["total_loss"].sum()).backward()
        self.trainer.scaler.step(self.trainer.optimizer)
        self.trainer.scaler.update()
        self.trainer.lr_scheduler_callback._scheduler.step()

    def _compare_optimizers(self, a, b):
        state_dict_a = a.state_dict()
        state_dict_b = b.state_dict()
        state_a = state_dict_a["state"]
        state_b = state_dict_b["state"]

        same = True
        same = same and list(state_a.keys()) == list(state_b.keys())
        same = same and state_dict_a["param_groups"] == state_dict_b["param_groups"]

        for item1, item2 in zip(state_a.values(), state_b.values()):
            same = same and compare_state_dicts(item1, item2)

        return same
