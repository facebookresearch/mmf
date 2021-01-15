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
from mmf.utils.general import get_max_updates, print_model_parameters
from pytorch_lightning import Trainer


logger = logging.getLogger(__name__)


@registry.register_trainer("lightning")
class LightningTrainer(BaseTrainer):
    def __init__(self, config: mmf_typings.DictConfig):
        super().__init__(config)
        self.trainer = None

    def load(self):
        super().load()
        self.trainer_config = self.config.trainer.params
        self._calculate_max_updates()
        self._load_trainer()

    def _load_trainer(self):
        lightning_params = self.trainer_config
        self.trainer = Trainer(
            logger=False,
            gpus=lightning_params.gpus,
            num_nodes=lightning_params.num_nodes,
            callbacks=self._callbacks,
            precision=lightning_params.precision,
            deterministic=lightning_params.deterministic,
            benchmark=lightning_params.benchmark,
            max_steps=self._max_updates,
            max_epochs=self._max_epochs,
            gradient_clip_val=lightning_params.gradient_clip_val,
            num_sanity_val_steps=lightning_params.num_sanity_val_steps,
            automatic_optimization=lightning_params.automatic_optimization,
            checkpoint_callback=lightning_params.checkpoint_callback,
            accumulate_grad_batches=lightning_params.accumulate_grad_batches,
            val_check_interval=lightning_params.val_check_interval,
            log_every_n_steps=lightning_params.log_every_n_steps,
        )

    def configure_device(self):
        logger.info("Configure device: noop for lightning")

    def configure_seed(self):
        logger.info("Configure seed: noop for lightning")

    def load_datasets(self):
        logger.info("Loading datasets")
        data_module = LightningDataModule(self.config)
        data_module.prepare_data()
        self.data_module = data_module

    def load_model(self):
        logger.info("Loading models")

        attributes = self.config.model_config[self.config.model]
        if isinstance(attributes, str):
            attributes = self.config.model_config[attributes]
        with omegaconf.open_dict(attributes):
            attributes.model = self.config.model

        self.model = build_model(attributes)
        self.model.is_pl_enabled = True

    def load_optimizer(self):
        logger.info("Loading optimizer: noop for lightning")

    def load_metrics(self) -> None:
        logger.info("Loading metrics")
        metrics = self.config.evaluation.get("metrics", [])
        self.metrics = Metrics(metrics)
        self.metrics_params = self.metrics.required_params

    def configure_callbacks(self) -> None:
        self._callbacks = [LightningLoopCallback(self)]

    def train(self):
        logger.info("===== Model =====")
        logger.info(self.model)
        print_model_parameters(self.model)

        logger.info("Starting training...")
        self.trainer.fit(self.model, self.data_module)

    def inference(self):
        logger.info("Starting inference...")
        # TODO: @sash coming soon
        pass

    def _calculate_max_updates(self):
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
