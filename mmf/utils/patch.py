# Copyright (c) Facebook, Inc. and its affiliates.

import importlib
import logging
import sys

from packaging import version


logger = logging.getLogger(__name__)


def patch_transformers(log_incompatible=False):
    import transformers

    if version.parse(transformers.__version__) < version.parse("4.0.0"):
        return
    if not hasattr(transformers, "models"):
        return

    logger.info(f"Patching transformers version: {transformers.__version__}")

    sys.path = sys.path[1:] + [sys.path[0]]

    for key in dir(transformers.models):
        if key.startswith("__"):
            continue

        model_lib = importlib.import_module(f"transformers.models.{key}")
        if not hasattr(model_lib, "_modules"):
            if log_incompatible:
                logger.info(
                    f"transformers' patching: model {key} has no "
                    + "_modules attribute. Skipping."
                )
            continue

        for module in model_lib._modules:
            if not module or module == ".":
                continue
            sys.modules[f"transformers.{module}"] = importlib.import_module(
                f"transformers.models.{key}.{module}"
            )
    sys.path = [sys.path[-1]] + sys.path[:-1]
