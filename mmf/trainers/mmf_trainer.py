# Copyright (c) Facebook, Inc. and its affiliates.

import omegaconf

from mmf.common import typings as mmf_typings
from mmf.common.dataset_loader import DatasetLoader
from mmf.common.registry import registry
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
from mmf.trainers.core.reporting import TrainerReportingMixin
from mmf.trainers.core.training_loop import TrainerTrainingLoopMixin
from mmf.utils.build import build_model, build_optimizer
from mmf.utils.general import print_model_parameters


@registry.register_trainer("mmf_trainer")
class MMFTrainer(
    TrainerCallbackHookMixin,
    TrainerTrainingLoopMixin,
    TrainerDeviceMixin,
    TrainerEvaluationLoopMixin,
    TrainerReportingMixin,
    TrainerProfilingMixin,
    BaseTrainer,
):
    def __init__(self, config: mmf_typings.DictConfig):
        super().__init__(config)

    def load(self):
        super().load()

        # Parallize model
        self.parallelize_model()

        # Callbacks
        self.on_init_end()

    def configure_callbacks(self):
        self.checkpoint_callback = CheckpointCallback(self.config, self)
        self.early_stop_callback = EarlyStoppingCallback(self.config, self)
        # self.callbacks.append(self.early_stop_callback)
        self.logistics_callback = LogisticsCallback(self.config, self)
        self.lr_scheduler_callback = LRSchedulerCallback(self.config, self)

        # Add callbacks for execution during events
        self.callbacks.append(self.checkpoint_callback)
        self.callbacks.append(self.logistics_callback)
        self.callbacks.append(self.lr_scheduler_callback)

    def load_datasets(self):
        self.writer.write("Loading datasets", "info")
        self.dataset_loader = DatasetLoader(self.config)
        self.dataset_loader.load_datasets()

        self.train_dataset = self.dataset_loader.train_dataset
        self.val_dataset = self.dataset_loader.val_dataset
        self.test_dataset = self.dataset_loader.test_dataset

        self.train_loader = self.dataset_loader.train_loader
        self.val_loader = self.dataset_loader.val_loader
        self.test_loader = self.dataset_loader.test_loader

    def load_model(self):
        self.writer.write("Loading model", "info")
        attributes = self.config.model_config[self.config.model]
        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_config[attributes]

        with omegaconf.open_dict(attributes):
            attributes.model = self.config.model

        self.model = build_model(attributes)
        self.model = self.model.to(self.device)

    def load_optimizer(self):
        self.writer.write("Loading optimizer", "info")
        self.optimizer = build_optimizer(self.model, self.config)

    def load_metrics(self) -> None:
        self.writer.write("Loading metrics", "info")
        metrics = self.config.evaluation.get("metrics", [])
        self.metrics = Metrics(metrics)
        self.metrics_params = self.metrics.required_params

    def train(self):
        self.writer.write("===== Model =====")
        self.writer.write(self.model)
        print_model_parameters(self.model)

        if "train" not in self.run_type:
            self.inference()
            return

        self.on_train_start()
        self.training_loop()
        self.on_train_end()

        self.inference()

    def inference(self):
        dataset_type = []
        if "val" in self.run_type:
            dataset_type.append("val")
        if any(rt in self.run_type for rt in ["inference", "test", "predict"]):
            dataset_type.append("test")

        for dataset in dataset_type:
            self.on_test_start()
            if self.config.evaluation.predict:
                self.prediction_loop(dataset)
                continue
            else:
                self.writer.write(f"Starting inference on {dataset} set")
                report, meter = self.evaluation_loop(
                    getattr(self, f"{dataset}_loader"), use_tqdm=True
                )
            self.on_test_end(report=report, meter=meter)
