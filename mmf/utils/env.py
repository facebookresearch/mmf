# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import importlib
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict


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


def import_user_module(user_dir: str):
    """Given a user dir, this function imports it as a module.

    This user_module is expected to have an __init__.py at its root.
    You can use import_files to import your python files easily in
    __init__.py

    Args:
        user_dir (str): directory which has to be imported
    """
    from mmf.common.registry import registry
    from mmf.utils.general import get_absolute_path  # noqa

    logger = logging.getLogger(__name__)
    if user_dir:
        if registry.get("__mmf_user_dir_imported__", no_warning=True):
            logger.info(f"User dir {user_dir} already imported. Skipping.")
            return

        # Allow loading of files as user source
        if user_dir.endswith(".py"):
            user_dir = user_dir[:-3]

        dot_path = ".".join(user_dir.split(os.path.sep))
        # In case of abspath which start from "/" the first char
        # will be "." which turns it into relative module which
        # find_spec doesn't like
        if os.path.isabs(user_dir):
            dot_path = dot_path[1:]

        try:
            dot_spec = importlib.util.find_spec(dot_path)
        except ModuleNotFoundError:
            dot_spec = None
        abs_user_dir = get_absolute_path(user_dir)
        module_parent, module_name = os.path.split(abs_user_dir)

        # If dot path is found in sys.modules, or path can be directly
        # be imported, we don't need to play jugglery with actual path
        if dot_path in sys.modules or dot_spec is not None:
            module_name = dot_path
        else:
            user_dir = abs_user_dir

        logger.info(f"Importing from {user_dir}")
        if module_name != dot_path:
            # Since dot path hasn't been found or can't be imported,
            # we can try importing the module by changing sys path
            # to the parent
            sys.path.insert(0, module_parent)

        importlib.import_module(module_name)
        sys.modules["mmf_user_dir"] = sys.modules[module_name]

        # Register config for user's model and dataset config
        # relative path resolution
        config = registry.get("config")
        if config is None:
            registry.register(
                "config", OmegaConf.create({"env": {"user_dir": user_dir}})
            )
        else:
            with open_dict(config):
                config.env.user_dir = user_dir

        registry.register("__mmf_user_dir_imported__", True)


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

        environment_mmf_path = os.environ.get("MMF_PATH", os.environ.get("PYTHIA_PATH"))

        if environment_mmf_path is not None:
            root_folder = environment_mmf_path

        registry.register("pythia_path", root_folder)
        registry.register("mmf_path", root_folder)

    trainer_folder = os.path.join(root_folder, "trainers")
    trainer_pattern = os.path.join(trainer_folder, "**", "*.py")
    datasets_folder = os.path.join(root_folder, "datasets")
    datasets_pattern = os.path.join(datasets_folder, "**", "*.py")
    model_folder = os.path.join(root_folder, "models")
    common_folder = os.path.join(root_folder, "common")
    modules_folder = os.path.join(root_folder, "modules")
    model_pattern = os.path.join(model_folder, "**", "*.py")
    common_pattern = os.path.join(common_folder, "**", "*.py")
    modules_pattern = os.path.join(modules_folder, "**", "*.py")

    importlib.import_module("mmf.common.meter")

    files = (
        glob.glob(datasets_pattern, recursive=True)
        + glob.glob(model_pattern, recursive=True)
        + glob.glob(trainer_pattern, recursive=True)
        + glob.glob(common_pattern, recursive=True)
        + glob.glob(modules_pattern, recursive=True)
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


def setup_torchaudio():
    # required for soundfile
    try:
        import libfb.py.ctypesmonkeypatch

        libfb.py.ctypesmonkeypatch.install()
    except ImportError:
        pass


def teardown_imports():
    from mmf.common.registry import registry

    registry.unregister("pythia_path")
    registry.unregister("mmf_path")
    registry.unregister("imports_setup")
