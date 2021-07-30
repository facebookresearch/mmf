# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
import os
from typing import Any, Dict, List

import omegaconf
from mmf.common.registry import registry
from mmf.datasets.multi_datamodule import MultiDataModule
from mmf.modules.metrics import Metrics
from mmf.trainers.base_trainer import BaseTrainer
from mmf.trainers.lightning_core.loop_callback import LightningLoopCallback
from mmf.utils.build import build_lightning_model
from mmf.utils.checkpoint import (
    get_ckpt_from_path,
    get_ckpt_path_from_folder,
    get_config_from_folder_or_ckpt,
)
from mmf.utils.checkpoint_updater import is_model_only_checkpoint
from mmf.utils.configuration import get_mmf_env
from mmf.utils.download import download_pretrained_model
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_max_updates, print_model_parameters
from mmf.utils.logger import TensorboardLogger, setup_output_folder
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


logger = logging.getLogger(__name__)


@registry.register_trainer("lightning")
class LightningTrainer(BaseTrainer):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.trainer = None
        self.trainer_config = self.config.trainer.params
        self.data_module = None
        self.resume_from_checkpoint = None

    def load(self):
        super().load()
        self._calculate_max_updates()
        self._load_loggers()
        self._load_trainer()

    def _load_trainer(self):
        lightning_params = self.trainer_config
        with omegaconf.open_dict(lightning_params):
            lightning_params.pop("max_steps")
            lightning_params.pop("max_epochs")
            lightning_params.pop("resume_from_checkpoint")

        lightning_params_dict = OmegaConf.to_container(lightning_params, resolve=True)
        self.trainer = Trainer(
            callbacks=self.callbacks,
            max_steps=self._max_updates,
            resume_from_checkpoint=self.resume_from_checkpoint,
            default_root_dir=get_mmf_env(key="log_dir"),
            **lightning_params_dict,
        )

    def configure_device(self) -> None:
        pass

    def configure_seed(self) -> None:
        seed = self.config.training.seed
        seed_everything(seed)

    def _load_loggers(self) -> None:
        self.tb_writer = None
        if self.training_config.tensorboard:
            # TODO: @sash PL logger upgrade
            log_dir = setup_output_folder(folder_only=True)
            env_tb_logdir = get_mmf_env(key="tensorboard_logdir")
            if env_tb_logdir:
                log_dir = env_tb_logdir

            self.tb_writer = TensorboardLogger(log_dir)

    def load_datasets(self) -> None:
        logger.info("Loading datasets")
        data_module = MultiDataModule(self.config)
        self.data_module = data_module

        self.train_loader = data_module.train_dataloader()
        self.val_loader = data_module.val_dataloader()
        self.test_loader = data_module.test_dataloader()

    def load_model(self) -> None:
        logger.info("Loading models")

        checkpoint_data = self.get_checkpoint_data()
        checkpoint_path = checkpoint_data["checkpoint_path"]
        ckpt = checkpoint_data["ckpt"]
        is_zoo = checkpoint_data["is_zoo"]
        config = checkpoint_data["config"]

        model_checkpoint_path = None
        if checkpoint_path is not None:
            assert ckpt, "checkpoint should have been loaded when path is available"
            if is_model_only_checkpoint(ckpt):
                # it is model only checkpoint, then we load it here
                model_checkpoint_path = checkpoint_path
            else:
                # it is a trainer checkpoint, we pass it as a trainer param
                self.resume_from_checkpoint = checkpoint_path

        attributes = self.get_model_config(is_zoo, config)
        self.model = build_lightning_model(attributes, model_checkpoint_path)
        self.model.build_meters(self.run_type)

    def get_model_config(
        self, is_zoo: bool = False, config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        ckpt_config = self.config.checkpoint
        if is_zoo and ckpt_config.zoo_config_override and config:
            self.config.model_config = config.model_config

        attributes = self.config.model_config[self.config.model]
        if isinstance(attributes, str):
            attributes = self.config.model_config[attributes]
        with omegaconf.open_dict(attributes):
            attributes.model = self.config.model

        return attributes

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """ This function gets checkpoint file path on disk from
        config.trainer.params.resume_from_checkpoint. However if it not specified,
        it gets checkpoint path from config.checkpoint. If config.resume is specified
        it gets the latest checkpoint from the config's save directory (alternatively it
        gets the best checkpoint if config.resume_best is True). If config.resume is not
        specified, then it gets config.resume_file or the checkpoint file from
        config.resume_zoo (in that order).

        Returns:
            Dict[str, Any]: a dict containing the following keys,
            `checkpoint_path` (str) local file path for the checkpoint;
            `ckpt` (Dict[str, Any])
            `is_zoo` (Bool) whether or not the checkpoint is specified through a
                zoo identifier
            `config` (Dict[str, Any]]) the config that is stored together with this
                checkpoint
        """
        # get ckpt file path from config.trainer.params.resume_from_checkpoint
        path = self.config.trainer.params.get("resume_from_checkpoint", None)
        if path is not None:
            is_zoo = self.is_zoo_path(path)
            ckpt_filepath = path
            if is_zoo:
                folder = download_pretrained_model(path)
                ckpt_filepath = get_ckpt_path_from_folder(folder)
                ckpt = get_ckpt_from_path(ckpt_filepath)
                config = get_config_from_folder_or_ckpt(folder, ckpt)
            else:
                ckpt = get_ckpt_from_path(ckpt_filepath)
                config = None

            return {
                "ckpt": ckpt,
                "checkpoint_path": ckpt_filepath,
                "is_zoo": is_zoo,
                "config": config,
            }

        is_zoo = False
        config = None
        ckpt = None
        # get ckpt file path from config.checkpoint
        ckpt_config = self.config.checkpoint
        suffix = "best.ckpt" if ckpt_config.resume_best else "current.ckpt"
        path = os.path.join(get_mmf_env(key="save_dir"), suffix)
        ckpt_filepath = None
        resume_from_specified_path = (
            ckpt_config.resume_file is not None or ckpt_config.resume_zoo is not None
        ) and (not ckpt_config.resume or not PathManager.exists(path))
        if resume_from_specified_path:
            if ckpt_config.resume_file and PathManager.exists(ckpt_config.resume_file):
                ckpt_filepath = ckpt_config.resume_file
            elif ckpt_config.resume_zoo is not None:
                is_zoo = True
                folder = download_pretrained_model(ckpt_config.resume_zoo)
                ckpt_filepath = get_ckpt_path_from_folder(folder)
                ckpt = get_ckpt_from_path(ckpt_filepath)
                config = get_config_from_folder_or_ckpt(folder, ckpt)
            else:
                raise RuntimeError(f"{ckpt_config.resume_file} doesn't exist")

        if ckpt_config.resume and PathManager.exists(path):
            ckpt_filepath = path

        if ckpt_filepath is not None:
            ckpt = get_ckpt_from_path(ckpt_filepath)

        return {
            "ckpt": ckpt,
            "checkpoint_path": ckpt_filepath,
            "is_zoo": is_zoo,
            "config": config,
        }

    def is_zoo_path(self, path) -> bool:
        from mmf.utils.configuration import get_mmf_env, load_yaml

        model_zoo = load_yaml(get_mmf_env(key="model_zoo"))
        OmegaConf.set_struct(model_zoo, True)
        OmegaConf.set_readonly(model_zoo, True)

        try:
            model_config = OmegaConf.select(model_zoo, path)
            return model_config is not None
        except omegaconf.errors.OmegaConfBaseException:
            return False

    def load_optimizer(self) -> None:
        logger.info("Loading optimizer: noop for lightning")

    def load_metrics(self) -> None:
        logger.info("Loading metrics")
        metrics = self.config.evaluation.get("metrics", [])
        # moved metrics into the model object
        self.model.metrics = Metrics(metrics)

    def monitor_criteria(self):
        monitor_criteria = self.training_config.early_stop.get("criteria", None)
        assert (
            monitor_criteria
        ), "monitor criteria is required when early stop is specified."
        if "val" not in monitor_criteria:
            monitor_criteria = f"val/{monitor_criteria}"
        mode = (
            "min" if self.training_config.early_stop.get("minimize", False) else "max"
        )
        return monitor_criteria, mode

    def configure_callbacks(self) -> None:
        self.callbacks = [LightningLoopCallback(self)]
        self.callbacks += self.configure_checkpoint_callbacks()
        if self.training_config.get(
            "early_stop", None
        ) and self.training_config.early_stop.get("enabled", False):
            self.callbacks += self.configure_monitor_callbacks()
            self.callbacks += self.configure_earlystop_callback()

    def configure_earlystop_callback(self) -> List[ModelCheckpoint]:
        return []

    def configure_checkpoint_callbacks(self) -> List[ModelCheckpoint]:
        train_callback = ModelCheckpoint(
            monitor=None,
            every_n_train_steps=self.config.training.checkpoint_interval,
            dirpath=get_mmf_env(key="save_dir"),
            filename="models/model_{step}",
            save_last=True,
            verbose=True,
        )
        train_callback.CHECKPOINT_NAME_LAST = "current"
        return [train_callback]

    def configure_monitor_callbacks(self) -> List[ModelCheckpoint]:
        criteria, mode = self.monitor_criteria()
        monitor_callback = ModelCheckpoint(
            monitor=criteria,
            dirpath=get_mmf_env(key="save_dir"),
            filename="best",
            mode=mode,
            save_top_k=1,
            save_last=False,
            verbose=True,
        )
        return [monitor_callback]

    def train(self) -> None:
        logger.info("===== Model =====")
        logger.info(self.model)
        print_model_parameters(self.model)

        logger.info("Starting training...")

        if "train" not in self.run_type:
            self.inference()
            return

        self.trainer.fit(self.model, self.data_module)
        self.run_last_validation_after_train()

        # TODO: Look for a better way to hook this
        self.data_module.teardown()

    def run_last_validation_after_train(self) -> None:
        # Don't run if current iteration is divisble by
        # val check interval as it will just be a repeat
        if (
            "val" in self.run_type
            and self.trainer.global_step % self.trainer_config.val_check_interval != 0
        ):
            logger.info("Stepping into final validation check")
            self.trainer.validate(self.model, self.val_loader)

    def inference(self) -> None:
        logger.info("Starting inference...")
        # TODO: @sash coming soon
        pass

    def _calculate_max_updates(self) -> None:
        self._max_updates = self.trainer_config.max_steps
        self._max_epochs = self.trainer_config.max_epochs
        if self._max_updates is None and self._max_epochs is None:
            raise ValueError("Neither max_updates nor max_epochs is specified.")

        self._max_updates, max_epochs = get_max_updates(
            self._max_updates,
            self._max_epochs,
            self.train_loader,
            self.trainer_config.accumulate_grad_batches,
        )
        self._max_epochs = math.ceil(max_epochs)
        return self._max_updates
