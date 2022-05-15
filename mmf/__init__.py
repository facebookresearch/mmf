# Copyright (c) Facebook, Inc. and its affiliates.
# isort:skip_file
# flake8: noqa: F401
from mmf.utils.patch import patch_transformers

patch_transformers()

from mmf import common, datasets, models, modules, utils
from mmf.modules import losses, metrics, optimizers, poolers, schedulers
from mmf.version import __version__


__all__ = [
    "utils",
    "common",
    "modules",
    "datasets",
    "models",
    "losses",
    "poolers",
    "schedulers",
    "optimizers",
    "metrics",
]
