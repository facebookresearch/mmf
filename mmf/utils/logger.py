# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import collections.abc
import functools
import json
import logging
import os
import sys
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

import torch
from mmf.common.registry import registry
from mmf.utils.configuration import get_mmf_env
from mmf.utils.distributed import get_rank, is_main, is_xla
from mmf.utils.file_io import PathManager
from mmf.utils.timer import Timer
from termcolor import colored


def setup_output_folder(folder_only: bool = False):
    """Sets up and returns the output file where the logs will be placed
    based on the configuration passed. Usually "save_dir/logs/log_<timestamp>.txt".
    If env.log_dir is passed, logs will be directly saved in this folder.

    Args:
        folder_only (bool, optional): If folder should be returned and not the file.
            Defaults to False.

    Returns:
        str: folder or file path depending on folder_only flag
    """
    save_dir = get_mmf_env(key="save_dir")
    time_format = "%Y_%m_%dT%H_%M_%S"
    log_filename = "train_"
    log_filename += Timer().get_time_hhmmss(None, format=time_format)
    log_filename += ".log"

    log_folder = os.path.join(save_dir, "logs")

    env_log_dir = get_mmf_env(key="log_dir")
    if env_log_dir:
        log_folder = env_log_dir

    if not PathManager.exists(log_folder):
        PathManager.mkdirs(log_folder)

    if folder_only:
        return log_folder

    log_filename = os.path.join(log_folder, log_filename)

    return log_filename


def setup_logger(
    output: str = None,
    color: bool = True,
    name: str = "mmf",
    disable: bool = False,
    clear_handlers=True,
    *args,
    **kwargs,
):
    """
    Initialize the MMF logger and set its verbosity level to "INFO".
    Outside libraries shouldn't call this in case they have set there
    own logging handlers and setup. If they do, and don't want to
    clear handlers, pass clear_handlers options.

    The initial version of this function was taken from D2 and adapted
    for MMF.

    Args:
        output (str): a file name or a directory to save log.
            If ends with ".txt" or ".log", assumed to be a file name.
            Default: Saved to file <save_dir/logs/log_[timestamp].txt>
        color (bool): If false, won't log colored logs. Default: true
        name (str): the root module name of this logger. Defaults to "mmf".
        clear_handlers (bool): If false, won't clear existing handlers.

    Returns:
        logging.Logger: a logger
    """
    if disable:
        return None
    logger = logging.getLogger(name)
    logger.propagate = False

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")

    plain_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s : %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    distributed_rank = get_rank()
    handlers = []

    config = registry.get("config")
    if config:
        logging_level = config.get("training", {}).get("logger_level", "info").upper()
    else:
        logging_level = logging.INFO

    if distributed_rank == 0:
        logger.setLevel(logging_level)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging_level)
        if color:
            formatter = ColorfulFormatter(
                colored("%(asctime)s | %(name)s: ", "green") + "%(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        warnings_logger.addHandler(ch)
        handlers.append(ch)

    # file logging: all workers
    if output is None:
        output = setup_output_folder()

    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "train.log")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging_level)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
        warnings_logger.addHandler(fh)
        handlers.append(fh)

        # Slurm/FB output, only log the main process
        if "train.log" not in filename and distributed_rank == 0:
            save_dir = get_mmf_env(key="save_dir")
            filename = os.path.join(save_dir, "train.log")
            sh = logging.StreamHandler(_cached_log_stream(filename))
            sh.setLevel(logging_level)
            sh.setFormatter(plain_formatter)
            logger.addHandler(sh)
            warnings_logger.addHandler(sh)
            handlers.append(sh)

        logger.info(f"Logging to: {filename}")

    # Remove existing handlers to add MMF specific handlers
    if clear_handlers:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    # Now, add our handlers.
    logging.basicConfig(level=logging_level, handlers=handlers)

    registry.register("writer", logger)

    return logger


def setup_very_basic_config(color=True):
    plain_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s : %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    if color:
        formatter = ColorfulFormatter(
            colored("%(asctime)s | %(name)s: ", "green") + "%(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = plain_formatter
    ch.setFormatter(formatter)
    # Setup a minimal configuration for logging in case something tries to
    # log a message even before logging is setup by MMF.
    logging.basicConfig(level=logging.INFO, handlers=[ch])


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "mmf"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


def summarize_report(
    current_iteration,
    num_updates,
    max_updates,
    meter,
    should_print=True,
    extra=None,
    tb_writer=None,
    wandb_logger=None,
):
    if extra is None:
        extra = {}
    if not is_main() and not is_xla():
        return

    # Log the learning rate if available
    if wandb_logger and "lr" in extra:
        wandb_logger.log_metrics(
            {"train/learning_rate": float(extra["lr"])}, commit=False
        )

    if tb_writer:
        scalar_dict = meter.get_scalar_dict()
        tb_writer.add_scalars(scalar_dict, current_iteration)

    if wandb_logger:
        metrics = meter.get_scalar_dict()
        wandb_logger.log_metrics({**metrics, "trainer/global_step": current_iteration})

    if not should_print:
        return
    log_dict = {}
    if num_updates is not None and max_updates is not None:
        log_dict.update({"progress": f"{num_updates}/{max_updates}"})

    log_dict.update(meter.get_log_dict())
    log_dict.update(extra)

    log_progress(log_dict)


def calculate_time_left(
    max_updates,
    num_updates,
    timer,
    num_snapshot_iterations,
    log_interval,
    eval_interval,
):
    if num_updates is None or max_updates is None:
        return "Unknown"

    time_taken_for_log = time.time() * 1000 - timer.start
    iterations_left = max_updates - num_updates
    num_logs_left = iterations_left / log_interval
    time_left = num_logs_left * time_taken_for_log

    snapshot_iteration = num_snapshot_iterations / log_interval
    if eval_interval:
        snapshot_iteration *= iterations_left / eval_interval
    time_left += snapshot_iteration * time_taken_for_log

    return timer.get_time_hhmmss(gap=time_left)


def log_progress(info: Union[Dict, Any], log_format="simple"):
    """Useful for logging progress dict.

    Args:
        info (dict|any): If dict, will be logged as key value pair. Otherwise,
            it will be logged directly.

        log_format (str, optional): json|simple. Defaults to "simple".
            Will use simple mode.
    """
    caller, key = _find_caller()
    logger = logging.getLogger(caller)

    if not isinstance(info, collections.abc.Mapping):
        logger.info(info)

    if log_format == "simple":
        config = registry.get("config")
        if config:
            log_format = config.training.log_format

    if log_format == "simple":
        output = ", ".join([f"{key}: {value}" for key, value in info.items()])
    elif log_format == "json":
        output = json.dumps(info)
    else:
        output = str(info)

    logger.info(output)


def log_class_usage(component_type, klass):
    """This function is used to log the usage of different MMF components."""
    identifier = "MMF"
    if klass and hasattr(klass, "__name__"):
        identifier += f".{component_type}.{klass.__name__}"
    torch._C._log_api_usage_once(identifier)


def skip_if_tensorboard_inactive(fn: Callable) -> Callable:
    """
    Checks whether summary writer is initialized and rank is 0 (main)
    Args:
        fn (Callable): Function which should be called based on whether
            tensorboard should log or not
    """

    @wraps(fn)
    def wrapped_fn(self, *args: Any, **kwargs: Any) -> Optional[Any]:
        if self.summary_writer is None or not self._is_main:
            return None
        else:
            return fn(self, *args, **kwargs)

    return wrapped_fn


# ColorfulFormatter is adopted from Detectron2 and adapted for MMF
class ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


class TensorboardLogger:
    def __init__(self, log_folder="./logs", iteration=0):
        self._summary_writer = None
        self._is_main = is_main()
        self.timer = Timer()
        self.log_folder = log_folder
        self.time_format = "%Y-%m-%dT%H:%M:%S"
        current_time = self.timer.get_time_hhmmss(None, format=self.time_format)
        self.tensorboard_folder = os.path.join(
            self.log_folder, f"tensorboard_{current_time}"
        )

    @property
    def summary_writer(self):
        # Only on rank zero
        if not self._is_main:
            return None

        if self._summary_writer is None:
            # This would handle warning of missing tensorboard
            from torch.utils.tensorboard import SummaryWriter

            self._summary_writer = SummaryWriter(self.tensorboard_folder)

        return self._summary_writer

    @skip_if_tensorboard_inactive
    def close(self):
        """
        Closes the tensorboard summary writer.
        """
        self.summary_writer.close()

    @skip_if_tensorboard_inactive
    def add_scalar(self, key, value, iteration):
        self.summary_writer.add_scalar(key, value, iteration)

    @skip_if_tensorboard_inactive
    def add_scalars(self, scalar_dict, iteration):
        for key, val in scalar_dict.items():
            self.summary_writer.add_scalar(key, val, iteration)

    @skip_if_tensorboard_inactive
    def add_histogram_for_model(self, model, iteration):
        for name, param in model.named_parameters():
            np_param = param.clone().cpu().data.numpy()
            self.summary_writer.add_histogram(name, np_param, iteration)


class WandbLogger:
    r"""
    Log using `Weights and Biases`.

    Args:
        entity: An entity is a username or team name where you're sending runs.
        config: Configuration for the run.
        project: Name of the W&B project.

    Raises:
        ImportError: If wandb package is not installed.
    """

    def __init__(
        self,
        entity: Optional[str] = None,
        config: Optional[Dict] = None,
        project: Optional[str] = None,
    ):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        self._wandb = wandb
        self._wandb_init = dict(entity=entity, config=config, project=project)
        wandb_kwargs = dict(config.training.wandb)
        wandb_kwargs.pop("enabled")
        wandb_kwargs.pop("entity")
        wandb_kwargs.pop("project")
        wandb_kwargs.pop("log_checkpoint")
        self._wandb_init.update(**wandb_kwargs)

        self.setup()

    def setup(self):
        """
        Setup `Weights and Biases` for logging.
        """
        if is_main():
            if self._wandb.run is None:
                self._wandb.init(**self._wandb_init)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("trainer/global_step")
                self._wandb.define_metric(
                    "*", step_metric="trainer/global_step", step_sync=True
                )

    def __del__(self):
        if getattr(self, "_wandb", None) is not None:
            self._wandb.finish()

    def _should_log_wandb(self):
        if self._wandb is None or not is_main():
            return False
        else:
            return True

    def log_metrics(self, metrics: Dict[str, float], commit=True):
        """
        Log the monitored metrics to the wand dashboard.

        Args:
            metrics (Dict[str, float]): A dictionary of metrics to log.
            commit (bool): Save the metrics dict to the wandb server and
                           increment the step. (default: True)
        """
        if not self._should_log_wandb():
            return

        self._wandb.log(metrics, commit=commit)

    def log_model_checkpoint(self, model_path):
        """
        Log the model checkpoint to the wandb dashboard.

        Args:
            model_path (str): Path to the model file.
        """
        if not self._should_log_wandb():
            return

        model_artifact = self._wandb.Artifact(
            "run_" + self._wandb.run.id + "_model", type="model"
        )

        model_artifact.add_file(model_path, name="current.pt")
        self._wandb.log_artifact(model_artifact, aliases=["latest"])
