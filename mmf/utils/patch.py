# Copyright (c) Facebook, Inc. and its affiliates.

import importlib
import logging
import sys

from packaging import version


logger = logging.getLogger(__name__)


def patch_transformers(log_incompatible=False):
    """
    Patches transformers version > 4.x to work with code that
    was written for version < 4.x. Specifically, before you
    could do something like `from transformers.modeling_bert import x`
    but this was moved to
    `from transformers.models.bert.modeling_bert import x`
    in newer versions. This functions fixes this discrepancy by adding
    these modules back to path.

    Another thing this function fixes is the conflict with local
    datasets folder vs huggingface datasets library in loading
    of transformers > 4.x version. To achieve this we modify sys.path
    to look for local folder at the last in path resolver. This is
    reverted back to original behavior at the end of the function.
    """
    import transformers

    # pl uses importlib to find_transformers spec throwing if None
    # this prevents mmf/__init__() from raising and value error
    if transformers.__spec__ is None:
        transformers.__spec__ = "MISSING"

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
            if not module or module == "." or module[0] == ".":
                continue
            sys.modules[f"transformers.{module}"] = importlib.import_module(
                f"transformers.models.{key}.{module}"
            )
    sys.path = [sys.path[-1]] + sys.path[:-1]
