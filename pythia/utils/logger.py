# Copyright (c) Facebook, Inc. and its affiliates.
import base64
import logging
import os
import sys

from tensorboardX import SummaryWriter

from pythia.utils.distributed_utils import is_main_process
from pythia.utils.general import (ckpt_name_from_core_args,
                                  foldername_from_config_override)
from pythia.utils.timer import Timer


class Logger:
    def __init__(self, config):
        self.logger = None
        self.summary_writer = None
        self._is_main_process = is_main_process()

        self.timer = Timer()
        self.config = config
        self.save_dir = config.training_parameters.save_dir
        self.log_folder = ckpt_name_from_core_args(config)
        self.log_folder += foldername_from_config_override(config)
        time_format = "%Y-%m-%dT%H:%M:%S"
        self.log_filename = ckpt_name_from_core_args(config) + "_"
        self.log_filename += self.timer.get_time_hhmmss(None, format=time_format)
        self.log_filename += ".log"

        self.log_folder = os.path.join(self.save_dir, self.log_folder, "logs")

        arg_log_dir = self.config.get("log_dir", None)
        if arg_log_dir:
            self.log_folder = arg_log_dir

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder, exist_ok=True)


        self.log_filename = os.path.join(self.log_folder, self.log_filename)

        if self._is_main_process:
            tensorboard_folder = os.path.join(self.log_folder, "tensorboard")
            self.summary_writer = SummaryWriter(tensorboard_folder)
            print("Logging to:", self.log_filename)

        logging.captureWarnings(True)

        self.logger = logging.getLogger(__name__)
        self._file_only_logger = logging.getLogger(__name__)
        warnings_logger = logging.getLogger("py.warnings")

        # Set level
        level = config["training_parameters"].get("logger_level", "info")
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

        should_not_log = self.config["training_parameters"]["should_not_log"]
        self.should_log = not should_not_log

        # Single log wrapper map
        self._single_log_map = set()

    def __del__(self):
        if getattr(self, "summary_writer", None) is not None:
            self.summary_writer.close()

    def write(self, x, level="info", donot_print=False, log_all=False):
        if self.logger is None:
            return

        if log_all is False and not self._is_main_process:
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

    def single_write(self, x, level="info"):
        if x + "_" + level in self._single_log_map:
            return
        else:
            self.write(x, level)

    def _should_log_tensorboard(self):
        if self.summary_writer is None:
            return False

        if not self._is_main_process:
            return False

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
