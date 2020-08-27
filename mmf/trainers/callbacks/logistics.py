# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import time

import torch
from mmf.trainers.callbacks.base import Callback
from mmf.utils.configuration import get_mmf_env
from mmf.utils.distributed import is_master
from mmf.utils.logger import TensorboardLogger, log_progress, setup_output_folder
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
        self.snapshot_iterations = len(self.trainer.val_dataset)
        self.snapshot_iterations //= self.training_config.batch_size

        self.tb_writer = None

        if self.training_config.tensorboard:
            log_dir = setup_output_folder(folder_only=True)
            env_tb_logdir = get_mmf_env(key="tensorboard_logdir")
            if env_tb_logdir:
                log_dir = env_tb_logdir

            self.tb_writer = TensorboardLogger(log_dir, self.trainer.current_iteration)

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

        extra.update(
            {
                "epoch": self.trainer.current_epoch,
                "num_updates": self.trainer.num_updates,
                "iterations": self.trainer.current_iteration,
                "max_updates": self.trainer.max_updates,
                "lr": "{:.5f}".format(
                    self.trainer.optimizer.param_groups[0]["lr"]
                ).rstrip("0"),
                "ups": "{:.2f}".format(
                    self.log_interval / self.train_timer.unix_time_since_start()
                ),
                "time": self.train_timer.get_time_since_start(),
                "time_since_start": self.total_timer.get_time_since_start(),
                "eta": self._calculate_time_left(),
            }
        )
        self.train_timer.reset()
        self._summarize_report(kwargs["meter"], extra=extra)

    def on_validation_start(self, **kwargs):
        self.snapshot_timer.reset()

    def on_validation_end(self, **kwargs):
        extra = {
            "num_updates": self.trainer.num_updates,
            "epoch": self.trainer.current_epoch,
            "iterations": self.trainer.current_iteration,
            "max_updates": self.trainer.max_updates,
            "val_time": self.snapshot_timer.get_time_since_start(),
        }
        extra.update(self.trainer.early_stop_callback.early_stopping.get_info())
        self.train_timer.reset()
        self._summarize_report(kwargs["meter"], extra=extra)

    def on_test_end(self, **kwargs):
        prefix = "{}: full {}".format(
            kwargs["report"].dataset_name, kwargs["report"].dataset_type
        )
        self._summarize_report(kwargs["meter"], prefix)
        logger.info(f"Finished run in {self.total_timer.get_time_since_start()}")

    def _summarize_report(self, meter, should_print=True, extra=None):
        if extra is None:
            extra = {}
        if not is_master():
            return

        if self.training_config.tensorboard:
            scalar_dict = meter.get_scalar_dict()
            self.tb_writer.add_scalars(scalar_dict, self.trainer.current_iteration)

        if not should_print:
            return
        log_dict = {}
        if hasattr(self.trainer, "num_updates") and hasattr(
            self.trainer, "max_updates"
        ):
            log_dict.update(
                {"progress": f"{self.trainer.num_updates}/{self.trainer.max_updates}"}
            )
        log_dict.update(meter.get_log_dict())
        log_dict.update(extra)

        log_progress(log_dict)

    def _calculate_time_left(self):
        time_taken_for_log = time.time() * 1000 - self.train_timer.start
        iterations_left = self.trainer.max_updates - self.trainer.num_updates
        num_logs_left = iterations_left / self.log_interval
        time_left = num_logs_left * time_taken_for_log

        snapshot_iteration = self.snapshot_iterations / self.log_interval
        snapshot_iteration *= iterations_left / self.evaluation_interval
        time_left += snapshot_iteration * time_taken_for_log

        return self.train_timer.get_time_hhmmss(gap=time_left)
