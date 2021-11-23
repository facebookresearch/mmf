# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import torch
from mmf.trainers.callbacks.base import Callback
from mmf.utils.configuration import get_mmf_env
from mmf.utils.logger import (
    TensorboardLogger,
    WandbLogger,
    calculate_time_left,
    setup_output_folder,
    summarize_report,
)
from mmf.utils.timer import Timer


logger = logging.getLogger(__name__)


class LogisticsCallback(Callback):
    """Callback for handling train/validation logistics, report summarization,
    logging etc.
    """

    def __init__(self, config, trainer):
        """
        Attr:
            config(mmf_typings.DictConfig): Config for the callback
            trainer(Type[BaseTrainer]): Trainer object
        """
        super().__init__(config, trainer)

        self.total_timer = Timer()
        self.log_interval = self.training_config.log_interval
        self.evaluation_interval = self.training_config.evaluation_interval
        self.checkpoint_interval = self.training_config.checkpoint_interval

        # Total iterations for snapshot
        # len would be number of batches per GPU == max updates
        self.snapshot_iterations = len(self.trainer.val_loader)

        self.tb_writer = None

        self.wandb_logger = None

        if self.training_config.tensorboard:
            log_dir = setup_output_folder(folder_only=True)
            env_tb_logdir = get_mmf_env(key="tensorboard_logdir")
            if env_tb_logdir:
                log_dir = env_tb_logdir

            self.tb_writer = TensorboardLogger(log_dir, self.trainer.current_iteration)

        if self.training_config.wandb.enabled:
            log_dir = setup_output_folder(folder_only=True)

            env_wandb_logdir = get_mmf_env(key="wandb_logdir")
            if env_wandb_logdir:
                log_dir = env_wandb_logdir

            self.wandb_logger = WandbLogger(
                entity=config.training.wandb.entity,
                config=config,
                project=config.training.wandb.project,
            )

    def on_train_start(self):
        self.train_timer = Timer()
        self.snapshot_timer = Timer()

    def on_update_end(self, **kwargs):
        if not kwargs["should_log"]:
            return
        extra = {}
        if "cuda" in str(self.trainer.device):
            extra["max mem"] = torch.cuda.max_memory_allocated() / 1024
            extra["max mem"] //= 1024

        if self.training_config.experiment_name:
            extra["experiment"] = self.training_config.experiment_name

        max_updates = getattr(self.trainer, "max_updates", None)
        num_updates = getattr(self.trainer, "num_updates", None)
        extra.update(
            {
                "epoch": self.trainer.current_epoch,
                "num_updates": num_updates,
                "iterations": self.trainer.current_iteration,
                "max_updates": max_updates,
                "lr": "{:.5f}".format(
                    self.trainer.optimizer.param_groups[0]["lr"]
                ).rstrip("0"),
                "ups": "{:.2f}".format(
                    self.log_interval / self.train_timer.unix_time_since_start()
                ),
                "time": self.train_timer.get_time_since_start(),
                "time_since_start": self.total_timer.get_time_since_start(),
                "eta": calculate_time_left(
                    max_updates=max_updates,
                    num_updates=num_updates,
                    timer=self.train_timer,
                    num_snapshot_iterations=self.snapshot_iterations,
                    log_interval=self.log_interval,
                    eval_interval=self.evaluation_interval,
                ),
            }
        )
        self.train_timer.reset()
        summarize_report(
            current_iteration=self.trainer.current_iteration,
            num_updates=num_updates,
            max_updates=max_updates,
            meter=kwargs["meter"],
            extra=extra,
            tb_writer=self.tb_writer,
            wandb_logger=self.wandb_logger,
        )

    def on_validation_start(self, **kwargs):
        self.snapshot_timer.reset()

    def on_validation_end(self, **kwargs):
        max_updates = getattr(self.trainer, "max_updates", None)
        num_updates = getattr(self.trainer, "num_updates", None)
        extra = {
            "num_updates": num_updates,
            "epoch": self.trainer.current_epoch,
            "iterations": self.trainer.current_iteration,
            "max_updates": max_updates,
            "val_time": self.snapshot_timer.get_time_since_start(),
        }
        extra.update(self.trainer.early_stop_callback.early_stopping.get_info())
        self.train_timer.reset()
        summarize_report(
            current_iteration=self.trainer.current_iteration,
            num_updates=num_updates,
            max_updates=max_updates,
            meter=kwargs["meter"],
            extra=extra,
            tb_writer=self.tb_writer,
            wandb_logger=self.wandb_logger,
        )

    def on_test_end(self, **kwargs):
        prefix = "{}: full {}".format(
            kwargs["report"].dataset_name, kwargs["report"].dataset_type
        )
        summarize_report(
            current_iteration=self.trainer.current_iteration,
            num_updates=getattr(self.trainer, "num_updates", None),
            max_updates=getattr(self.trainer, "max_updates", None),
            meter=kwargs["meter"],
            should_print=prefix,
            tb_writer=self.tb_writer,
            wandb_logger=self.wandb_logger,
        )
        logger.info(f"Finished run in {self.total_timer.get_time_since_start()}")

    def teardown(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
