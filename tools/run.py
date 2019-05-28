# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import importlib
import os

from pythia.common.registry import registry
from pythia.utils.build_utils import build_trainer
from pythia.utils.distributed_utils import is_main_process
from pythia.utils.flags import flags


def setup_imports():
    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("pythia_root", no_warning=True)

    if root_folder is None:
        root_folder = os.path.dirname(os.path.abspath(__file__))
        root_folder = os.path.join(root_folder, "..")

        environment_pythia_path = os.environ.get("PYTHIA_PATH")

        if environment_pythia_path is not None:
            root_folder = environment_pythia_path

        root_folder = os.path.join(root_folder, "pythia")
        registry.register("pythia_path", root_folder)

    trainer_folder = os.path.join(root_folder, "trainers")
    trainer_pattern = os.path.join(trainer_folder, "**", "*.py")
    tasks_folder = os.path.join(root_folder, "tasks")
    tasks_pattern = os.path.join(tasks_folder, "**", "*.py")
    model_folder = os.path.join(root_folder, "models")
    model_pattern = os.path.join(model_folder, "**", "*.py")

    importlib.import_module("pythia.common.meter")

    files = glob.glob(tasks_pattern, recursive=True) + \
            glob.glob(model_pattern, recursive=True) + \
            glob.glob(trainer_pattern, recursive=True)

    for f in files:
        if f.endswith("task.py"):
            splits = f.split(os.sep)
            task_name = splits[-2]
            if task_name == "tasks":
                continue
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("pythia.tasks." + task_name + "." + module_name)
        elif f.find("models") != -1:
            splits = f.split(os.sep)
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("pythia.models." + module_name)
        elif f.find("trainer") != -1:
            splits = f.split(os.sep)
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("pythia.trainers." + module_name)
        elif f.endswith("builder.py"):
            splits = f.split(os.sep)
            task_name = splits[-3]
            dataset_name = splits[-2]
            if task_name == "tasks" or dataset_name == "tasks":
                continue
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module(
                "pythia.tasks." + task_name + "." + dataset_name + "." + module_name
            )


def run():
    setup_imports()
    parser = flags.get_parser()
    args = parser.parse_args()
    trainer = build_trainer(args)

    # Log any errors that occur to log file
    try:
        trainer.load()
        trainer.train()
    except Exception as e:
        writer = getattr(trainer, "writer", None)

        if writer is not None:
            writer.write(e, "error", donot_print=True)
        if is_main_process():
            raise


if __name__ == "__main__":
    run()
