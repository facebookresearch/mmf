# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
import os
from typing import Any, List, Optional

import omegaconf
from mmf.common.registry import registry
from mmf.datasets.multi_datamodule import MultiDataModule
from mmf.modules.metrics import Metrics
from mmf.trainers.base_trainer import BaseTrainer
from mmf.trainers.lightning_core.loop_callback import LightningLoopCallback
from mmf.utils.build import build_model
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

        lightning_params_dict = OmegaConf.to_container(lightning_params, resolve=True)
        self.trainer = Trainer(
            callbacks=self._callbacks,
            max_steps=self._max_updates,
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

        attributes = self.config.model_config[self.config.model]
        if isinstance(attributes, str):
            attributes = self.config.model_config[attributes]
        with omegaconf.open_dict(attributes):
            attributes.model = self.config.model

        checkpoint_path = self.get_checkpoint_path()
        self.model = build_model(attributes, checkpoint_path, True)
        self.model.build_meters(self.run_type)

    def get_checkpoint_path(self) -> Optional[str]:
        ckpt_config = self.config.checkpoint
        suffix = "best.ckpt" if ckpt_config.resume_best else "current.ckpt"
        default_ckpt_filepath = os.path.join(get_mmf_env(key="save_dir"), suffix)

        ckpt_filepath = None
        resume_from_specified_path = (
            ckpt_config.resume_file is not None or ckpt_config.resume_zoo is not None
        ) and (not ckpt_config.resume or not PathManager.exists(default_ckpt_filepath))
        if resume_from_specified_path:
            if ckpt_config.resume_file and PathManager.exists(ckpt_config.resume_file):
                ckpt_filepath = ckpt_config.resume_file
            elif ckpt_config.resume_zoo is not None:
                ckpt_filepath = download_pretrained_model(ckpt_config.resume_zoo)
            else:
                raise RuntimeError(f"{ckpt_config.resume_file} doesn't exist")

        if ckpt_config.resume and PathManager.exists(default_ckpt_filepath):
            ckpt_filepath = default_ckpt_filepath

        return ckpt_filepath

    def load_optimizer(self) -> None:
        logger.info("Loading optimizer: noop for lightning")

    def load_metrics(self) -> None:
        logger.info("Loading metrics")
        metrics = self.config.evaluation.get("metrics", [])
        # moved metrics into the model object
        self.model.metrics = Metrics(metrics)

    def monitor_criteria(self):
        monitor_criteria = self.training_config.early_stop.criteria
        if "val" not in monitor_criteria:
            monitor_criteria = f"val/{monitor_criteria}"
        return monitor_criteria

    def configure_callbacks(self) -> None:
        self._callbacks = [LightningLoopCallback(self)]
        # TODO @sash: add early stop callback
        # earlystop_callback = self.configure_earlystop_callback()
        checkpoint_callbacks = self.configure_checkpoint_callbacks()
        self._callbacks += checkpoint_callbacks

    def configure_earlystop_callback(self) -> List[Any]:
        pass

    def configure_checkpoint_callbacks(self) -> List[Any]:
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
