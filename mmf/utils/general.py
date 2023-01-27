# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import gc
import logging
import math
import os
import sys
import time
import warnings
from bisect import bisect
from typing import Any, Callable, Dict

import torch
from mmf.utils.distributed import get_rank, get_world_size, is_xla
from mmf.utils.file_io import PathManager
from packaging import version
from torch import nn, Tensor


logger = logging.getLogger(__name__)


def lr_lambda_update(i_iter, cfg):
    if cfg.training.use_warmup is True and i_iter <= cfg.training.warmup_iterations:
        alpha = float(i_iter) / float(cfg.training.warmup_iterations)
        return cfg.training.warmup_factor * (1.0 - alpha) + alpha
    else:
        idx = bisect(cfg.training.lr_steps, i_iter)
        return pow(cfg.training.lr_ratio, idx)


def clip_gradients(model, optimizer, i_iter, writer, config, scale=1.0):
    max_grad_l2_norm = config.training.max_grad_l2_norm
    clip_norm_mode = config.training.clip_norm_mode

    if max_grad_l2_norm is not None:
        if clip_norm_mode == "all":
            if hasattr(optimizer, "clip_grad_norm"):
                norm = optimizer.clip_grad_norm(max_grad_l2_norm * scale)
            else:
                norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_l2_norm * scale
                )
            if writer is not None:
                writer.add_scalars({"grad_norm": norm}, i_iter)
        else:
            raise NotImplementedError(
                "Clip norm mode %s not implemented" % clip_norm_mode
            )


def ckpt_name_from_core_args(config):
    seed = config.training.seed

    ckpt_name = f"{config.datasets}_{config.model}"

    if seed is not None:
        ckpt_name += f"_{seed:d}"

    return ckpt_name


def foldername_from_config_override(args):
    cfg_override = None
    if hasattr(args, "config_override"):
        cfg_override = args.config_override
    elif "config_override" in args:
        cfg_override = args["config_override"]

    folder_name = ""
    if cfg_override is not None and len(cfg_override) > 0:
        folder_name = str(cfg_override)
        folder_name = folder_name.replace(":", ".").replace("\n", " ")
        folder_name = folder_name.replace("/", "_")
        folder_name = " ".join(folder_name.split())
        folder_name = folder_name.replace(". ", ".").replace(" ", "_")
        folder_name = "_" + folder_name
    return folder_name


def get_mmf_root():
    from mmf.common.registry import registry

    mmf_root = registry.get("mmf_root", no_warning=True)
    if mmf_root is None:
        mmf_root = os.path.dirname(os.path.abspath(__file__))
        mmf_root = os.path.abspath(os.path.join(mmf_root, ".."))
        registry.register("mmf_root", mmf_root)
    return mmf_root


def get_absolute_path(paths):
    # String check should be first as Sequence would pass for string too
    if isinstance(paths, str):
        # If path is absolute return it directly
        if os.path.isabs(paths):
            return paths

        possible_paths = [
            # Direct path
            paths
        ]
        # Now, try relative to user_dir if it exists
        from mmf.utils.configuration import get_mmf_env

        mmf_root = get_mmf_root()
        user_dir = get_mmf_env(key="user_dir")
        if user_dir:
            possible_paths.append(os.path.join(user_dir, paths))
            # check in relative to mmf relative user dir
            possible_paths.append(os.path.join(mmf_root, "..", user_dir, paths))

        # Relative to root folder of mmf install
        possible_paths.append(os.path.join(mmf_root, "..", paths))
        # Relative to mmf root
        possible_paths.append(os.path.join(mmf_root, paths))

        # Test all these paths, if any exists return
        for path in possible_paths:
            if PathManager.exists(path):
                # URIs
                if path.find("://") == -1:
                    return os.path.abspath(path)
                else:
                    return path

        # If nothing works, return original path so that it throws an error
        return paths
    elif isinstance(paths, collections.abc.Iterable):
        return [get_absolute_path(path) for path in paths]
    else:
        raise TypeError("Paths passed to dataset should either be " "string or list")


def get_optimizer_parameters(model, config):
    parameters = model.parameters()

    has_custom = hasattr(model, "get_optimizer_parameters")
    if has_custom:
        parameters = model.get_optimizer_parameters(config)

    is_parallel = isinstance(model, nn.DataParallel) or isinstance(
        model, nn.parallel.DistributedDataParallel
    )

    if is_parallel and hasattr(model.module, "get_optimizer_parameters"):
        parameters = model.module.get_optimizer_parameters(config)

    # If parameters are a generator, convert to a list first
    parameters = list(parameters)

    if len(parameters) == 0:
        raise ValueError("optimizer got an empty parameter list")

    # If parameters are in format of list, instead of grouped params
    # convert them to grouped params form
    if not isinstance(parameters[0], dict):
        parameters = [{"params": parameters}]

    for group in parameters:
        group["params"] = list(group["params"])

    check_unused_parameters(parameters, model, config)

    return parameters


def check_unused_parameters(parameters, model, config):
    optimizer_param_set = {p for group in parameters for p in group["params"]}
    unused_param_names = []
    for n, p in model.named_parameters():
        if p.requires_grad and p not in optimizer_param_set:
            unused_param_names.append(n)
    if len(unused_param_names) > 0:
        logger.info(
            "Model parameters not used by optimizer: {}".format(
                " ".join(unused_param_names)
            )
        )
        if not config.optimizer.allow_unused_parameters:
            raise Exception(
                "Found model parameters not used by optimizer. Please check the "
                "model's get_optimizer_parameters and add all parameters. If this "
                "is intended, set optimizer.allow_unused_parameters to True to "
                "ignore it."
            )


def dict_to_string(dictionary):
    logs = []
    if dictionary is None:
        return ""
    for key, val in dictionary.items():
        if hasattr(val, "item"):
            val = val.item()
        # if key.count('_') == 2:
        #     key = key[key.find('_') + 1:]
        logs.append(f"{key}: {val:.4f}")

    return ", ".join(logs)


def get_overlap_score(candidate, target):
    """Takes a candidate word and a target word and returns the overlap
    score between the two.

    Parameters
    ----------
    candidate : str
        Candidate word whose overlap has to be detected.
    target : str
        Target word against which the overlap will be detected

    Returns
    -------
    float
        Overlap score betwen candidate and the target.

    """
    if len(candidate) < len(target):
        temp = candidate
        candidate = target
        target = temp
    overlap = 0.0
    while len(target) >= 2:
        if target in candidate:
            overlap = len(target)
            return overlap * 1.0 / len(candidate)
        else:
            target = target[:-1]
    return 0.0


def updir(d, n):
    """Given path d, go up n dirs from d and return that path"""
    ret_val = d
    for _ in range(n):
        ret_val = os.path.dirname(ret_val)
    return ret_val


def print_cuda_usage():
    print("Memory Allocated:", torch.cuda.memory_allocated() / (1024 * 1024))
    print("Max Memory Allocated:", torch.cuda.max_memory_allocated() / (1024 * 1024))
    print("Memory Cached:", torch.cuda.memory_cached() / (1024 * 1024))
    print("Max Memory Cached:", torch.cuda.max_memory_cached() / (1024 * 1024))


def check_fft_version():
    # Acquires and parses the PyTorch version
    if version.parse(torch.__version__) >= version.parse("1.7"):
        if "torch.fft" not in sys.modules:
            raise RuntimeError("torch.fft module available but not imported")


def rfft(input_tensor, signal_ndim=1, n=None, dim=-1, norm=None) -> torch.Tensor:
    check_fft_version()
    if "torch.fft" not in sys.modules:
        return torch.rfft(input_tensor, signal_ndim=signal_ndim)
    else:
        return torch.fft.rfft(input_tensor, n, dim, norm)


def irfft(input_tensor, s=None, signal_ndim=1, dim=None, norm=None) -> torch.Tensor:
    check_fft_version()
    if "torch.fft" not in sys.modules:
        return torch.irfft(input_tensor, signal_ndim=signal_ndim, signal_sizes=s)
    else:
        return torch.fft.irfftn(input_tensor, s, dim, norm)


def get_current_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(type(obj), obj.size())
        except Exception:
            pass


def get_batch_size():
    from mmf.utils.configuration import get_global_config

    batch_size = get_global_config("training.batch_size")
    world_size = get_world_size()

    batch_size_per_device = get_global_config("training.batch_size_per_device")

    if batch_size_per_device is not None:
        logger.info(
            f"training.batch_size_per_device has been used as {batch_size_per_device} "
            + "This will override training.batch_size and set the global batch size to "
            + f"{batch_size_per_device} x {world_size} = "
            + f"{batch_size_per_device * world_size}"
        )
        batch_size = batch_size_per_device * world_size

    if batch_size % world_size != 0:
        raise RuntimeError(
            "Batch size {} must be divisible by number "
            "of GPUs {} used.".format(batch_size, world_size)
        )

    return batch_size // world_size


def print_model_parameters(model, return_only=False):
    total_params = sum(p.numel() for p in model.parameters())
    trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if not return_only:
        logger.info(
            f"Total Parameters: {total_params}. Trained Parameters: {trained_params}"
        )
    return total_params, trained_params


def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)  # Adjust last
    assert sum(sizes_list) == dim
    if sizes_list[-1] < 0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j - 1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list


def get_max_updates(config_max_updates, config_max_epochs, train_loader, update_freq):
    if config_max_updates is None and config_max_epochs is None:
        raise ValueError("Neither max_updates nor max_epochs is specified.")

    if isinstance(train_loader.current_dataset, torch.utils.data.IterableDataset):
        warnings.warn(
            "max_epochs not supported for Iterable datasets. Falling back "
            + "to max_updates."
        )
        return config_max_updates, config_max_epochs

    if config_max_updates is not None and config_max_epochs is not None:
        warnings.warn(
            "Both max_updates and max_epochs are specified. "
            + f"Favoring max_epochs: {config_max_epochs}"
        )

    if config_max_epochs is not None:
        assert (
            hasattr(train_loader, "__len__") and len(train_loader) != 0
        ), "max_epochs can't be used with IterableDatasets"
        max_updates = math.ceil(len(train_loader) / update_freq) * config_max_epochs
        max_epochs = config_max_epochs
    else:
        max_updates = config_max_updates
        if hasattr(train_loader, "__len__") and len(train_loader) != 0:
            max_epochs = max_updates / len(train_loader)
        else:
            max_epochs = math.inf

    return max_updates, max_epochs


def extract_loss(report: Dict[str, Any], loss_divisor: int) -> torch.Tensor:
    loss_dict = report.losses
    assert len(loss_dict) != 0, (
        "Model returned an empty loss dict. "
        "Did you forget to (i) define losses in your model configuration or"
        "(ii) return losses dict from your model?"
    )

    # Since losses are batch averaged in MMF, this makes sure the
    # scaling is right.
    for key, value in loss_dict.items():
        value = value.mean() / loss_divisor
        report.losses[key] = value

    loss = sum(loss.mean() for loss in loss_dict.values())
    return loss


def get_chunks(x, sizes):
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1, begin, s)
        out.append(y)
        begin += s
    return out


def filter_grads(parameters):
    return [param for param in parameters if param.requires_grad]


def log_device_names():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        logger.info(f"CUDA Device {get_rank()} is: {device_name}")


def assert_iterator_finished(iter):
    try:
        _ = next(iter)
    except StopIteration:
        pass
    else:
        assert False


def get_current_device():
    if is_xla():
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return torch.device("cpu")


def retry_n(n: int, fn: Callable, *args, log_tries=False, **kwargs) -> Any:
    """Retries a function n times with increasing exponentionally
    increasing sleep intervals in between. First argument is number of tries
    if n==1, means function will be called at least twice, first is try, second
    is retry. Second argument is the function itself, rest of the arguments and
    keyword arguments are passed to the function directly. Returns the output
    of the function directly. if failed after n retries, the exception will be
    raised.

    Args:
        n (int): Number of tries to be made
        fn (Callable): Function to be called
        log_tries (bool): If the function should log the try iteration. Default: False

    Returns:
        Any: Output from fn
    """
    completed = False
    count = 0
    output = None

    while not completed:
        try:
            output = fn(*args, **kwargs)
            completed = True
        except Exception:
            if count < n:
                if log_tries:
                    logger.info(
                        f"Try {count + 1}/{n} failed for {fn.__name__}. Will retry "
                        f"after {2 ** count} second(s)."
                    )
                time.sleep(2**count)
                count += 1
            else:
                raise

    return output


def scalarize_dict_values(dict_with_tensors: Dict[str, Tensor]):
    """
    this method returns a new dict where the values of
    `dict_with_tensors` would be a scalar

    Returns:
        Dict: a new dict with scalarized values
    """
    dict_with_scalar_tensors = {}
    for key, val in dict_with_tensors.items():
        if torch.is_tensor(val):
            if val.dim() != 0:
                val = val.mean()
        dict_with_scalar_tensors[key] = val
    return dict_with_scalar_tensors
