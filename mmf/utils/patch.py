# Copyright (c) Facebook, Inc. and its affiliates.

import importlib
import logging
import sys

from mmf.common.registry import registry
from packaging import version


logger = logging.getLogger(__name__)
ORIGINAL_PATCH_FUNCTIONS_KEY = "original_patch_functions"
registry.register(ORIGINAL_PATCH_FUNCTIONS_KEY, {})


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
    try:
        import transformers3 as transformers
    except ImportError:
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


def safecopy_modules(module_function_names, caller_modules):
    """
    Saves a reference to each module.function in list of strings module_function_names.
    References are made from dict caller_modules, from module name str to
    caller module obj.
    module.functions can be reassigned, replacing the current functions using
    restore_saved_modules(caller_modules)

    Example:
        from transformers.modeling_bert import BertSelfAttention

        caller_modules = {'BertSelfAttention': BertSelfAttention}
        original_forward = BertSelfAttention.forward
        safecopy_modules(['BertSelfAttention.forward'], caller_modules)
        BertSelfAttention.forward = None
        restore_saved_modules(caller_modules)
        assert( original_forward is BertSelfAttention.forward )
    """
    original_functions = registry.get(ORIGINAL_PATCH_FUNCTIONS_KEY)
    for module_function_name in module_function_names:
        module_name, function_name = module_function_name.split(".")
        module = caller_modules[module_name]
        function = getattr(module, function_name)

        # store function is nothing is stored,
        # prevents multiple calls from overwriting original function
        original_functions[module_function_name] = original_functions.get(
            module_function_name, function
        )


def restore_saved_modules(caller_globals):
    """
    Restore function for safecopy_modules()
    Reassigns current dictionary of 'module.function': function
    saved by safecopy_modules to callers modules.
    Assumes caller_globals is a dict from module name str to caller module obj.

    Example:
        restore_saved_modules({'BertSelfAttention': BertSelfAttention})
    """
    original_functions = registry.get(ORIGINAL_PATCH_FUNCTIONS_KEY)
    for module_function_name, function in original_functions.items():
        module_name, function_name = module_function_name.split(".")
        if module_name in caller_globals:
            setattr(caller_globals[module_name], function_name, function)
    registry.register(ORIGINAL_PATCH_FUNCTIONS_KEY, {})
