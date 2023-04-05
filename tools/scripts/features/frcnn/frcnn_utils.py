# Copyright (c) Facebook, Inc. and its affiliates.

"""
 coding=utf-8
 Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal, Huggingface team :)
 Adapted From Facebook Inc, Detectron2
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.import copy
 """

import copy
import json
import os
import pickle as pkl
import shutil
import tarfile
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse
from zipfile import is_zipfile, ZipFile

import numpy as np
import requests
from filelock import FileLock
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.download import download
from omegaconf import OmegaConf


mmf_cache_home = get_mmf_cache_dir()


try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False


default_cache_path = os.path.join(mmf_cache_home, "transformers")

PATH = "/".join(str(Path(__file__).resolve()).split("/")[:-1])
CONFIG = os.path.join(PATH, "config.yaml")
ATTRIBUTES = os.path.join(PATH, "attributes.txt")
OBJECTS = os.path.join(PATH, "objects.txt")
PYTORCH_PRETRAINED_BERT_CACHE = os.getenv(
    "PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path
)
PYTORCH_TRANSFORMERS_CACHE = os.getenv(
    "PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE
)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)
WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.yaml"


def load_labels(objs=OBJECTS, attrs=ATTRIBUTES):
    vg_classes = []
    with open(objs) as f:
        for object in f.readlines():
            vg_classes.append(object.split(",")[0].lower().strip())

    vg_attrs = []
    with open(attrs) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(",")[0].lower().strip())
    return vg_classes, vg_attrs


def load_checkpoint(ckp):
    r = OrderedDict()
    with open(ckp, "rb") as f:
        ckp = pkl.load(f)["model"]
    for k in copy.deepcopy(list(ckp.keys())):
        v = ckp.pop(k)
        if isinstance(v, np.ndarray):
            v = torch.tensor(v)
        else:
            assert isinstance(v, torch.tensor), type(v)
        r[k] = v
    return r


class Config:
    _pointer = {}

    def __init__(self, dictionary: dict, name: str = "root", level=0):
        self._name = name
        self._level = level
        d = {}
        for k, v in dictionary.items():
            if v is None:
                raise ValueError()
            k = copy.deepcopy(k)
            v = copy.deepcopy(v)
            if isinstance(v, dict):
                v = Config(v, name=k, level=level + 1)
            d[k] = v
            setattr(self, k, v)

        self._pointer = d

    def __repr__(self):
        return str(list(self._pointer.keys()))

    def __setattr__(self, key, val):
        self.__dict__[key] = val
        self.__dict__[key.upper()] = val
        levels = key.split(".")
        last_level = len(levels) - 1
        pointer = self._pointer
        if len(levels) > 1:
            for i, level in enumerate(levels):
                if hasattr(self, level) and isinstance(getattr(self, level), Config):
                    setattr(getattr(self, level), ".".join(levels[i:]), val)
                if level == last_level:
                    pointer[level] = val
                else:
                    pointer = pointer[level]

    def to_dict(self):
        return self._pointer

    def dump_yaml(self, data, file_name):
        with open(f"{file_name}", "w") as stream:
            OmegaConf.save(data, stream)

    def dump_json(self, data, file_name):
        with open(f"{file_name}", "w") as stream:
            json.dump(data, stream)

    @staticmethod
    def load_yaml(config):
        return dict(OmegaConf.load(config))

    def __str__(self):
        t = "    "
        if self._name != "root":
            r = f"{t * (self._level-1)}{self._name}:\n"
        else:
            r = ""
        level = self._level
        for i, (k, v) in enumerate(self._pointer.items()):
            if isinstance(v, Config):
                r += f"{t * (self._level)}{v}\n"
                self._level += 1
            else:
                r += f"{t * (self._level)}{k}: {v} ({type(v).__name__})\n"
            self._level = level
        return r[:-1]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(config_dict)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)

        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path

        try:
            # Load from URL or cache if already cached
            resolved_config_file = cached_path(
                config_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
            )
            # Load config dict
            if resolved_config_file is None:
                raise OSError

            config_file = Config.load_yaml(resolved_config_file)

        except OSError:
            msg = "Can't load config for"
            raise OSError(msg)

        if resolved_config_file == config_file:
            print("loading configuration file from path")
        else:
            print("loading configuration file cache")

        return Config.load_yaml(resolved_config_file), kwargs


# quick compare tensors
def compare(in_tensor):
    out_tensor = torch.load("dump.pt", map_location=in_tensor.device)
    n1 = in_tensor.numpy()
    n2 = out_tensor.numpy()[0]
    print(n1.shape, n1[0, 0, :5])
    print(n2.shape, n2[0, 0, :5])
    summ = sum([1 for x in np.isclose(n1, n2, rtol=0.01, atol=0.1).flatten() if not x])
    assert np.allclose(
        n1, n2, rtol=0.01, atol=0.1
    ), f"{summ/len(n1.flatten())*100:.4f} % element-wise mismatch"
    raise Exception("tensors are all good")

    # Hugging face functions below


# TODO use mmf's download functionality
def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent=None,
    extract_compressed_file=False,
    force_extract=False,
    local_files_only=False,
):
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_remote_url(url_or_filename):
        filename = url_to_filename(url_or_filename)
        output_path = os.path.join(cache_dir, filename)
        if not os.path.isfile(output_path):
            assert download(
                url_or_filename, cache_dir, filename
            ), f"failed to download {url_or_filename}"
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise OSError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(
            f"unable to parse {url_or_filename} as a URL or as a local path"
        )

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted"
        # at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if (
            os.path.isdir(output_path_extracted)
            and os.listdir(output_path_extracted)
            and not force_extract
        ):
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise OSError(
                    f"Archive format of {output_path} could not be identified"
                )

        return output_path_extracted

    return output_path


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def url_to_filename(url, etag=None):
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"
    elif url.endswith(".yaml"):
        filename += ".yaml"
    elif url.endswith(".bin"):
        filename += ".bin"

    return filename


def get_data(query, delim=","):
    assert isinstance(query, str)
    if os.path.isfile(query):
        with open(query) as f:
            data = eval(f.read())
    else:
        req = requests.get(query)
        try:
            data = requests.json()
        except Exception:
            data = req.content.decode()
            assert data is not None, "could not connect"
            try:
                data = eval(data)
            except Exception:
                data = data.split("\n")
        req.close()
    return data
