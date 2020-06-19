# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import importlib
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

from mmf.utils.general import get_absolute_path


def set_seed(seed):
    if seed:
        if seed == -1:
            # From detectron2
            seed = (
                os.getpid()
                + int(datetime.now().strftime("%S%f"))
                + int.from_bytes(os.urandom(2), "big")
            )
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    return seed


def import_user_module(user_dir: str, no_print: bool = False):
    """Given a user dir, this function imports it as a module.

    This user_module is expected to have an __init__.py at its root.
    You can use import_files to import your python files easily in
    __init__.py

    Args:
        user_dir (str): directory which has to be imported
        no_print (bool): This function won't print anything if set to true
    """
    if user_dir:
        user_dir = get_absolute_path(user_dir)
        module_parent, module_name = os.path.split(user_dir)

        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            if not no_print:
                print(f"Importing user_dir from {user_dir}")
            importlib.import_module(module_name)
            sys.path.pop(0)


def import_files(file_path: str, module_name: str = None):
    """The function imports all of the files present in file_path's directory.
    This is useful for end user in case they want to easily import files without
    mentioning each of them in their __init__.py. module_name if specified
    is the full path to module under which all modules will be imported.

    my_project/
        my_models/
            my_model.py
            __init__.py

    Contents of __init__.py

    ```
    from mmf.utils.env import import_files

    import_files(__file__, "my_project.my_models")
    ```

    This will then allow you to import `my_project.my_models.my_model` anywhere.

    Args:
        file_path (str): Path to file in whose directory everything will be imported
        module_name (str): Module name if this file under some specified structure
    """
    for file in os.listdir(os.path.dirname(file_path)):
        if file.endswith(".py") and not file.startswith("_"):
            import_name = file[: file.find(".py")]
            if module_name:
                importlib.import_module(f"{module_name}.{import_name}")
            else:
                importlib.import_module(f"{import_name}")


def setup_imports():
    from mmf.common.registry import registry

    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return
    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("mmf_root", no_warning=True)

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
        f = os.path.realpath(f)
        if f.endswith(".py") and not f.endswith("__init__.py"):
            splits = f.split(os.sep)
            import_prefix_index = 0
            for idx, split in enumerate(splits):
                if split == "mmf":
                    import_prefix_index = idx + 1
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            module = ".".join(["mmf"] + splits[import_prefix_index:-1] + [module_name])
            importlib.import_module(module)

    registry.register("imports_setup", True)
