# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import json
import logging
import os
import sys

from torch.utils.tensorboard import SummaryWriter

from mmf.utils.distributed import is_master
from mmf.utils.timer import Timer


class Logger:
    def __init__(self, config):
        self.logger = None
        self._is_master = is_master()

        self.timer = Timer()
        self.config = config
        self.save_dir = config.training.save_dir
        self.log_format = config.training.log_format
        self.time_format = "%Y-%m-%dT%H:%M:%S"
        self.log_filename = "train_"
        self.log_filename += self.timer.get_time_hhmmss(None, format=self.time_format)
        self.log_filename += ".log"

        self.log_folder = os.path.join(self.save_dir, "logs")

        if "log_dir" in self.config and self.config.log_dir is not None:
            self.log_folder = self.config.log_dir

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder, exist_ok=True)

        self.log_filename = os.path.join(self.log_folder, self.log_filename)

        if not self._is_master:
            return
        if self._is_master:
            print("Logging to:", self.log_filename)

        logging.captureWarnings(True)

        self.logger = logging.getLogger(__name__)
        self._file_only_logger = logging.getLogger(__name__)
        warnings_logger = logging.getLogger("py.warnings")

        # Set level
        level = config.training.logger_level
        self.logger.setLevel(getattr(logging, level.upper()))
        self._file_only_logger.setLevel(getattr(logging, level.upper()))

        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )

        # Add handler to file
        channel = logging.FileHandler(filename=self.log_filename, mode="a")
        channel.setFormatter(formatter)

        self.logger.addHandler(channel)
        self._file_only_logger.addHandler(channel)
        warnings_logger.addHandler(channel)

        # Add handler to stdout
        channel = logging.StreamHandler(sys.stdout)
        channel.setFormatter(formatter)

        self.logger.addHandler(channel)
        warnings_logger.addHandler(channel)

        should_not_log = self.config.training.should_not_log
        self.should_log = not should_not_log

        # Single log wrapper map
        self._single_log_map = set()

    def write(self, x, level="info", donot_print=False, log_all=False):
        if self.logger is None:
            return

        if log_all is False and not self._is_master:
            return

        # if it should not log then just print it
        if self.should_log:
            if hasattr(self.logger, level):
                if donot_print:
                    getattr(self._file_only_logger, level)(str(x))
                else:
                    getattr(self.logger, level)(str(x))
            else:
                self.logger.error("Unknown log level type: %s" % level)
        else:
            print(str(x) + "\n")

    def log_progress(self, info):
        if not isinstance(info, collections.Mapping):
            self.write(info)

        if not self._is_master:
            return

        if self.log_format == "simple":
            output = ", ".join(
                ["{}: {}".format(key, value) for key, value in info.items()]
            )
        elif self.log_format == "json":
            output = json.dumps(info)
        else:
            output = str(info)

        self.write(output)

    def single_write(self, x, level="info", log_all=False):
        if self.logger is None:
            return
        if log_all is False and not self._is_master:
            return
        if x + "_" + level in self._single_log_map:
            return
        else:
            self.write(x, level)


class TensorboardLogger:
    def __init__(self, log_folder="./logs", iteration=0):
        self.summary_writer = None
        self._is_master = is_master()
        self.timer = Timer()
        self.log_folder = log_folder
        self.time_format = "%Y-%m-%dT%H:%M:%S"

        if self._is_master:
            current_time = self.timer.get_time_hhmmss(None, format=self.time_format)
            tensorboard_folder = os.path.join(
                self.log_folder, "tensorboard_{}".format(current_time)
            )
            self.summary_writer = SummaryWriter(tensorboard_folder)

    def __del__(self):
        if getattr(self, "summary_writer", None) is not None:
            self.summary_writer.close()

    def _should_log_tensorboard(self):
        if self.summary_writer is None or not self._is_master:
            return False
        else:
            return True

    def add_scalar(self, key, value, iteration):
        if not self._should_log_tensorboard():
            return

        self.summary_writer.add_scalar(key, value, iteration)

    def add_scalars(self, scalar_dict, iteration):
        if not self._should_log_tensorboard():
            return

        for key, val in scalar_dict.items():
            self.summary_writer.add_scalar(key, val, iteration)

    def add_histogram_for_model(self, model, iteration):
        if not self._should_log_tensorboard():
            return

        for name, param in model.named_parameters():
            np_param = param.clone().cpu().data.numpy()
            self.summary_writer.add_histogram(name, np_param, iteration)
