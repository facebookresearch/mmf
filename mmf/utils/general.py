# Copyright (c) Facebook, Inc. and its affiliates.

import gc
import glob
import importlib
import os
import tarfile
import zipfile
from bisect import bisect

import requests
import torch
import tqdm
import yaml
from torch import nn

from mmf.common.constants import DOWNLOAD_CHUNK_SIZE
from mmf.utils.distributed import get_world_size


def lr_lambda_update(i_iter, cfg):
    if cfg.training.use_warmup is True and i_iter <= cfg.training.warmup_iterations:
        alpha = float(i_iter) / float(cfg.training.warmup_iterations)
        return cfg.training.warmup_factor * (1.0 - alpha) + alpha
    else:
        idx = bisect(cfg.training.lr_steps, i_iter)
        return pow(cfg.training.lr_ratio, idx)


def clip_gradients(model, i_iter, writer, config):
    # TODO: Fix question model retrieval
    max_grad_l2_norm = config.training.max_grad_l2_norm
    clip_norm_mode = config.training.clip_norm_mode

    if max_grad_l2_norm is not None:
        if clip_norm_mode == "all":
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_l2_norm)
            if writer is not None:
                writer.add_scalars({"grad_norm": norm}, i_iter)

        elif clip_norm_mode == "question":
            question_embedding = model.module.question_embedding_module
            norm = nn.utils.clip_grad_norm(
                question_embedding.parameters(), max_grad_l2_norm
            )
            if writer is not None:
                writer.add_scalars({"question_grad_norm": norm}, i_iter)
        else:
            raise NotImplementedError(
                "Clip norm mode %s not implemented" % clip_norm_mode
            )


def ckpt_name_from_core_args(config):
    seed = config.training.seed

    ckpt_name = "{}_{}".format(config.datasets, config.model)

    if seed is not None:
        ckpt_name += "_{:d}".format(seed)

    return ckpt_name


def foldername_from_config_override(args):
    cfg_override = None
    if hasattr(args, "config_override"):
        cfg_override = args.config_override
    elif "config_override" in args:
        cfg_override = args["config_override"]

    folder_name = ""
    if cfg_override is not None and len(cfg_override) > 0:
        folder_name = yaml.safe_dump(cfg_override, default_flow_style=True)
        folder_name = folder_name.replace(":", ".").replace("\n", " ")
        folder_name = folder_name.replace("/", "_")
        folder_name = " ".join(folder_name.split())
        folder_name = folder_name.replace(". ", ".").replace(" ", "_")
        folder_name = "_" + folder_name
    return folder_name


def get_mmf_root():
    from mmf.common.registry import registry

    pythia_root = registry.get("pythia_root", no_warning=True)
    if pythia_root is None:
        pythia_root = os.path.dirname(os.path.abspath(__file__))
        pythia_root = os.path.abspath(os.path.join(pythia_root, ".."))
        registry.register("pythia_root", pythia_root)
    return pythia_root


def get_mmf_cache_dir():
    from mmf.common.registry import registry

    config = registry.get("config")
    cache_dir = config.mmf_config.cache_dir
    # If cache_dir path exists do not join to mmf root
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(get_mmf_root(), cache_dir)
    return cache_dir


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

    return parameters


def dict_to_string(dictionary):
    logs = []
    if dictionary is None:
        return ""
    for key, val in dictionary.items():
        if hasattr(val, "item"):
            val = val.item()
        # if key.count('_') == 2:
        #     key = key[key.find('_') + 1:]
        logs.append("%s: %.4f" % (key, val))

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
    from mmf.common.registry import registry

    batch_size = registry.get("config").training.batch_size

    world_size = get_world_size()

    if batch_size % world_size != 0:
        raise RuntimeError(
            "Batch size {} must be divisible by number "
            "of GPUs {} used.".format(batch_size, world_size)
        )

    return batch_size // world_size


def print_model_parameters(model, return_only=False):
    from mmf.common.registry import registry

    writer = registry.get("writer")
    total_params = sum(p.numel() for p in model.parameters())
    trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if not return_only:
        writer.write(
            "Total Parameters: {}. Trained Parameters: {}".format(
                total_params, trained_params
            )
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


def get_chunks(x, sizes):
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1, begin, s)
        out.append(y)
        begin += s
    return out


def setup_imports():
    from mmf.common.registry import registry

    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("pythia_root", no_warning=True)

    if root_folder is None:
        root_folder = os.path.dirname(os.path.abspath(__file__))
        root_folder = os.path.join(root_folder, "..")

        environment_pythia_path = os.environ.get("PYTHIA_PATH")

        if environment_pythia_path is not None:
            root_folder = environment_pythia_path

        registry.register("pythia_path", root_folder)

    trainer_folder = os.path.join(root_folder, "trainers")
    trainer_pattern = os.path.join(trainer_folder, "**", "*.py")
    datasets_folder = os.path.join(root_folder, "datasets")
    datasets_pattern = os.path.join(datasets_folder, "**", "*.py")
    model_folder = os.path.join(root_folder, "models")
    model_pattern = os.path.join(model_folder, "**", "*.py")

    importlib.import_module("mmf.common.meter")

    files = (
        glob.glob(datasets_pattern, recursive=True)
        + glob.glob(model_pattern, recursive=True)
        + glob.glob(trainer_pattern, recursive=True)
    )

    for f in files:
        if f.find("models") != -1:
            splits = f.split(os.sep)
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("mmf.models." + module_name)
        elif f.find("trainer") != -1:
            splits = f.split(os.sep)
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("mmf.trainers." + module_name)
        elif f.endswith("builder.py"):
            splits = f.split(os.sep)
            task_name = splits[-3]
            dataset_name = splits[-2]
            if task_name == "datasets" or dataset_name == "datasets":
                continue
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module(
                "mmf.datasets." + task_name + "." + dataset_name + "." + module_name
            )
