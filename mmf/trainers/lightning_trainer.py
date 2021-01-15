# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math

import omegaconf
from mmf.common import typings as mmf_typings
from mmf.common.registry import registry
from mmf.datasets.lightning_datamodule import LightningDataModule
from mmf.modules.metrics import Metrics
from mmf.trainers.base_trainer import BaseTrainer
from mmf.trainers.lightning_core.loop_callback import LightningLoopCallback
from mmf.utils.build import build_model
from mmf.utils.configuration import get_mmf_env
from mmf.utils.flow import get_max_updates
from mmf.utils.general import print_model_parameters
from mmf.utils.logger import TensorboardLogger, setup_output_folder
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything


logger = logging.getLogger(__name__)


@registry.register_trainer("lightning")
class LightningTrainer(BaseTrainer):
    def __init__(self, config: mmf_typings.DictConfig):
        super().__init__(config)
        self.trainer = None
        self.trainer_config = self.config.trainer.params
        self.data_module = None

    def load(self):
        super().load()
        self._calculate_max_updates()
        loggers = self._load_loggers()
        self._load_trainer(loggers)

    def _load_trainer(self, loggers=False):
        lightning_params = self.trainer_config

        with omegaconf.open_dict(lightning_params):
            lightning_params.pop("max_steps")
            lightning_params.pop("max_epochs")

        lightning_params_dict = OmegaConf.to_container(lightning_params, resolve=True)
        self.trainer = Trainer(
            logger=False,
            callbacks=self._callbacks,
            max_steps=self._max_updates,
            default_root_dir=get_mmf_env(key="log_dir"),
            **lightning_params_dict
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
        return self.trainer_config.logger

    def load_datasets(self) -> None:
        logger.info("Loading datasets")
        data_module = LightningDataModule(self.config)
        self.data_module = data_module

    def load_model(self) -> None:
        logger.info("Loading models")

        attributes = self.config.model_config[self.config.model]
        if isinstance(attributes, str):
            attributes = self.config.model_config[attributes]
        with omegaconf.open_dict(attributes):
            attributes.model = self.config.model

        self.model = build_model(attributes)
        self.model.is_pl_enabled = True

    def load_optimizer(self) -> None:
        logger.info("Loading optimizer: noop for lightning")

    def load_metrics(self) -> None:
        logger.info("Loading metrics")
        metrics = self.config.evaluation.get("metrics", [])
        self.metrics = Metrics(metrics)
        self.metrics_params = self.metrics.required_params

    def configure_callbacks(self) -> None:
        self._callbacks = [LightningLoopCallback(self)]

    def train(self) -> None:
        logger.info("===== Model =====")
        logger.info(self.model)
        print_model_parameters(self.model)

        logger.info("Starting training...")

        if "train" not in self.run_type:
            self.inference()
            return

        self.trainer.fit(self.model, self.data_module)

    def inference(self) -> None:
        logger.info("Starting inference...")
        # TODO: @sash coming soon
        pass

    def _calculate_max_updates(self) -> None:
        self._max_updates = self.trainer_config.max_steps
        self._max_epochs = self.trainer_config.max_epochs
        if self._max_updates is None and self._max_epochs is None:
            raise ValueError("Neither max_updates nor max_epochs is specified.")

        train_loader = self.data_module.train_loader
        self._max_updates, max_epochs = get_max_updates(
            self._max_updates,
            self._max_epochs,
            train_loader,
            self.trainer_config.accumulate_grad_batches,
        )
        self._max_epochs = math.ceil(max_epochs)
        return self._max_updates
