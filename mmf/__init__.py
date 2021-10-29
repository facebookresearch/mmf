# Copyright (c) Facebook, Inc. and its affiliates.
# isort:skip_file
# flake8: noqa: F401
from mmf.utils.patch import patch_transformers

patch_transformers()

from mmf import utils, common, modules, datasets, models
from mmf.modules import losses, schedulers, optimizers, metrics, poolers
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
