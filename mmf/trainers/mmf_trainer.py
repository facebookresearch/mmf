# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import warnings

import omegaconf
import torch
from mmf.common.registry import registry
from mmf.datasets.multi_datamodule import MultiDataModule
from mmf.modules.metrics import Metrics
from mmf.trainers.base_trainer import BaseTrainer
from mmf.trainers.callbacks.checkpoint import CheckpointCallback
from mmf.trainers.callbacks.early_stopping import EarlyStoppingCallback
from mmf.trainers.callbacks.logistics import LogisticsCallback
from mmf.trainers.callbacks.lr_scheduler import LRSchedulerCallback
from mmf.trainers.core.callback_hook import TrainerCallbackHookMixin
from mmf.trainers.core.device import TrainerDeviceMixin
from mmf.trainers.core.evaluation_loop import TrainerEvaluationLoopMixin
from mmf.trainers.core.profiling import TrainerProfilingMixin
from mmf.trainers.core.training_loop import TrainerTrainingLoopMixin
from mmf.utils.build import build_model, build_optimizer
from mmf.utils.general import print_model_parameters
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


@registry.register_trainer("mmf")
class MMFTrainer(
    TrainerCallbackHookMixin,
    TrainerTrainingLoopMixin,
    TrainerDeviceMixin,
    TrainerEvaluationLoopMixin,
    TrainerProfilingMixin,
    BaseTrainer,
):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def load(self):
        super().load()
        self.load_fp16_scaler()

        # Callbacks
        self.on_init_start()

        # Parallize model
        self.parallelize_model()

        # Callbacks
        self.on_init_end()

    def configure_callbacks(self):
        self.checkpoint_callback = CheckpointCallback(self.config, self)
        self.early_stop_callback = EarlyStoppingCallback(self.config, self)
        self.logistics_callback = LogisticsCallback(self.config, self)
        self.lr_scheduler_callback = LRSchedulerCallback(self.config, self)

        # Add callbacks for execution during events
        self.callbacks.append(self.lr_scheduler_callback)
        # checkpoint_callback needs to be called after lr_scheduler_callback so that
        # lr_scheduler_callback._scheduler.step() happens before saving checkpoints
        # (otherwise the saved last_epoch in scheduler would be wrong)
        self.callbacks.append(self.checkpoint_callback)
        self.callbacks.append(self.logistics_callback)

    def load_datasets(self):
        logger.info("Loading datasets")
        self.dataset_loader = MultiDataModule(self.config)

        self.train_loader = self.dataset_loader.train_dataloader()
        self.val_loader = self.dataset_loader.val_dataloader()
        self.test_loader = self.dataset_loader.test_dataloader()

    def load_model(self):
        logger.info("Loading model")
        if self.config.model in self.config.model_config:
            attributes = self.config.model_config[self.config.model]
        else:
            warnings.warn(
                f"Model {self.config.model}'s config not present. "
                + "Continuing with empty config"
            )
            attributes = OmegaConf.create()
        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_config[attributes]

        with omegaconf.open_dict(attributes):
            attributes.model = self.config.model

        self.model = build_model(attributes)
        self.model = self.model.to(self.device)

    def load_optimizer(self):
        logger.info("Loading optimizer")
        self.optimizer = build_optimizer(self.model, self.config)

    def load_metrics(self) -> None:
        logger.info("Loading metrics")
        metrics = self.config.evaluation.get("metrics", [])
        self.metrics = Metrics(metrics)
        self.metrics_params = self.metrics.required_params

    def load_fp16_scaler(self):
        if self.training_config.fp16:
            assert (
                torch.__version__ >= "1.6"
            ), "Using fp16 requires torch version >- 1.6"
            assert self.device != torch.device("cpu"), "fp16 cannot be used on cpu"

        set_torch_grad_scaler = True
        if self.training_config.fp16 and self.distributed:
            try:
                from fairscale.optim.oss import OSS
                from fairscale.optim.grad_scaler import ShardedGradScaler

                if isinstance(self.optimizer, OSS):
                    self.scaler = ShardedGradScaler()
                    set_torch_grad_scaler = False
                    logger.info("Using FairScale ShardedGradScaler")
            except ImportError:
                logger.info("Using Pytorch AMP GradScaler")

        if set_torch_grad_scaler:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.training_config.fp16)

    def train(self):
        logger.info("===== Model =====")
        logger.info(self.model)
        print_model_parameters(self.model)

        if "train" not in self.run_type:
            self.inference()
            return

        self.on_train_start()
        self.training_loop()
        self.on_train_end()

        self.inference()
        self.dataset_loader.teardown()

    def inference(self):
        dataset_type = []
        if "val" in self.run_type:
            dataset_type.append("val")
        if any(rt in self.run_type for rt in ["inference", "test", "predict"]):
            dataset_type.append("test")

        for dataset in dataset_type:
            if self.config.evaluation.predict:
                self.on_prediction_start()
                self.prediction_loop(dataset)
                self.on_prediction_end()
            else:
                self.on_test_start()
                logger.info(f"Starting inference on {dataset} set")
                report, meter = self.evaluation_loop(dataset, use_tqdm=True)
                self.on_test_end(report=report, meter=meter)
