# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math

import omegaconf
import torch
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
        self._calculate_max_updates()
        self._calculate_gradient_clip_val()
        self._load_trainer()

    def _load_trainer(self):
        self.trainer = Trainer(
            logger=False,
            gpus=self._gpus,
            num_nodes=self._num_nodes,
            callbacks=self._callbacks,
            precision=16 if self.training_config.fp16 else 32,
            deterministic=self._deterministic,
            benchmark=self._benchmark,
            max_steps=self._max_updates,
            max_epochs=self._max_epochs,
            gradient_clip_val=self._gradient_clip_val,
            num_sanity_val_steps=0,
            automatic_optimization=True,
            checkpoint_callback=False,
            accumulate_grad_batches=self.training_config.update_frequency,
            val_check_interval=self.training_config.evaluation_interval,
            log_every_n_steps=self.training_config.log_interval,
        )

    def configure_device(self):
        # TODO: @sash coming soon!
        local_rank = self.config.device_id
        registry.register("global_device", local_rank)
        if self.config.distributed.init_method is not None:
            self._distributed = True
            self._gpus = [local_rank]
            self.device = torch.device("cuda", local_rank)
            # TODO: @sash set the corresponding node number
            assert 0, "Please set the corresponding nodes"
        elif torch.cuda.is_available():
            self._gpus = [0]
            self._num_nodes = 1
        else:
            self._gpus = None
            self._num_nodes = 1

    def configure_seed(self):
        default_deterministic = False
        seed = self.config.training.seed

        self._deterministic = default_deterministic
        if seed is not None:
            self._deterministic = True

        self._benchmark = False

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
        logger.info("Loading optimizer")
        if self.trainer:
            pass
            # TODO: sash, this is a problem, needs fix
            # at this point the model's configure_optimizer
            # function isnt called yet, so this will break.
            # self.optimizer = self.trainer.optimizers[0]
            # we need an assigned optimizer for checkpoint
            # to work

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
        self._max_updates = self.training_config.max_updates
        self._max_epochs = self.training_config.max_epochs
        if self._max_updates is None and self._max_epochs is None:
            raise ValueError("Neither max_updates nor max_epochs is specified.")

        train_loader = self.data_module.train_loader
        self._max_updates, max_epochs = get_max_updates(
            self._max_updates, self._max_epochs, train_loader
        )
        self._max_epochs = math.ceil(max_epochs)
        return self._max_updates

    def _calculate_gradient_clip_val(self):
        if not self.config.training.clip_gradients:
            self._gradient_clip_val = 0.0
            return

        max_grad_l2_norm = self.config.training.max_grad_l2_norm
        clip_norm_mode = self.config.training.clip_norm_mode

        if max_grad_l2_norm is not None:
            if clip_norm_mode == "all":
                self._gradient_clip_val = max_grad_l2_norm
            else:
                raise NotImplementedError(
                    "Clip norm mode %s not implemented" % clip_norm_mode
                )
