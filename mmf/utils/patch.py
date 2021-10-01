# Copyright (c) Facebook, Inc. and its affiliates.

import importlib
import logging
import sys

from packaging import version


logger = logging.getLogger(__name__)
original_functions = {}


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


def safecopy_modules(module_function_names, caller_globals):
    """
    Saves a reference to each module.function in list of strings module_function_names.
    References are made from global symbol table caller_globals.
    module.functions can be reassigned, replacing the current functions using
    restore_saved_modules(caller_globals)

    Example:
        from transformers.modeling_bert import BertSelfAttention

        original_forward = BertSelfAttention.forward
        safecopy_modules(['BertSelfAttention.forward'], globals())
        BertSelfAttention.forward = None
        restore_saved_modules(globals())
        assert( original_forward is BertSelfAttention.forward )
    """
    for module_function_name in module_function_names:
        module_name, function_name = module_function_name.split(".")
        module = caller_globals[module_name]
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
    Assumes caller_globals is the caller's global symbol table.

    Example:
        restore_saved_modules(global())
    """
    global original_functions
    for module_function_name, function in original_functions.items():
        module_name, function_name = module_function_name.split(".")
        if module_name in caller_globals:
            setattr(caller_globals[module_name], function_name, function)
    original_functions = {}
