# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import gc
import os
from bisect import bisect

import requests
import torch
import tqdm
import yaml
import tarfile
import zipfile

from torch import nn

from pythia.common.constants import DOWNLOAD_CHUNK_SIZE
from pythia.utils.distributed_utils import get_world_size


def lr_lambda_update(i_iter, cfg):
    if (
        cfg["training_parameters"]["use_warmup"] is True
        and i_iter <= cfg["training_parameters"]["warmup_iterations"]
    ):
        alpha = float(i_iter) / float(cfg["training_parameters"]["warmup_iterations"])
        return cfg["training_parameters"]["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        idx = bisect(cfg["training_parameters"]["lr_steps"], i_iter)
        return pow(cfg["training_parameters"]["lr_ratio"], idx)


def clip_gradients(model, i_iter, writer, config):
    # TODO: Fix question model retrieval
    max_grad_l2_norm = config["training_parameters"]["max_grad_l2_norm"]
    clip_norm_mode = config["training_parameters"]["clip_norm_mode"]

    if max_grad_l2_norm is not None:
        if clip_norm_mode == "all":
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_l2_norm)

            writer.add_scalars({"grad_norm": norm}, i_iter)

        elif clip_norm_mode == "question":
            question_embedding = model.module.question_embedding_module
            norm = nn.utils.clip_grad_norm(
                question_embedding.parameters(), max_grad_l2_norm
            )

            writer.add_scalars({"question_grad_norm": norm}, i_iter)
        else:
            raise NotImplementedError(
                "Clip norm mode %s not implemented" % clip_norm_mode
            )


def ckpt_name_from_core_args(config):
    seed = config["training_parameters"]["seed"]

    ckpt_name = "{}_{}".format(
        config["datasets"],
        config["model"]
    )

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


def get_pythia_root():
    from pythia.common.registry import registry

    pythia_root = registry.get("pythia_root", no_warning=True)
    if pythia_root is None:
        pythia_root = os.path.dirname(os.path.abspath(__file__))
        pythia_root = os.path.abspath(os.path.join(pythia_root, ".."))
        registry.register("pythia_root", pythia_root)
    return pythia_root


def download_file(url, output_dir=".", filename=""):
    if len(filename) == 0:
        filename = os.path.join(".", url.split("/")[-1])

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, filename)
    r = requests.get(url, stream=True)

    if r.status_code != requests.codes['ok']:
        print("The url {} is broken. If this is not your own url,"
              " please open up an issue on GitHub.".format(url))
    file_size = int(r.headers["Content-Length"])
    num_bars = int(file_size / DOWNLOAD_CHUNK_SIZE)

    with open(filename, "wb") as fh:
        for chunk in tqdm.tqdm(
            r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE),
            total=num_bars,
            unit="MB",
            desc=filename,
            leave=True,
        ):
            fh.write(chunk)


def get_optimizer_parameters(model, config):
    parameters = model.parameters()

    has_custom = hasattr(model, "get_optimizer_parameters")
    if has_custom:
        parameters = model.get_optimizer_parameters(config)

    is_parallel = (isinstance(model, nn.DataParallel) or
        isinstance(model, nn.parallel.DistributedDataParallel))

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
        except:
            pass


def extract_file(path, output_dir="."):
    _FILETYPE_TO_OPENER_MODE_MAPPING = {
        ".zip": (zipfile.ZipFile, "r"),
        ".tar.gz": (tarfile.open, "r:gz"),
        ".tgz": (tarfile.open, "r:gz"),
        ".tar": (tarfile.open, "r:"),
        ".tar.bz2": (tarfile.open, "r:bz2"),
        ".tbz": (tarfile.open, "r:bz2"),
    }

    cwd = os.getcwd()
    os.chdir(output_dir)

    extension = "." + ".".join(os.path.abspath(path).split(".")[1:])

    opener, mode = _FILETYPE_TO_OPENER_MODE_MAPPING[extension]
    with opener(path, mode) as f:
        f.extractall()

    os.chdir(cwd)
    return output_dir

def get_batch_size():
    from pythia.common.registry import registry

    batch_size = registry.get("config").training_parameters.batch_size

    world_size = get_world_size()

    if batch_size % world_size != 0:
        raise RuntimeError(
            "Batch size {} must be divisible by number "
            "of GPUs {} used.".format(batch_size, world_size)
        )

    return batch_size // world_size
