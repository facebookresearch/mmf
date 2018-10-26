import os
import importlib
import glob
from pythia.core.trainer import Trainer


def setup_imports():
    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = os.path.dirname(os.path.dirname(__file__))
    tasks_folder = os.path.join(root_folder, "tasks")
    tasks_pattern = os.path.join(tasks_folder, "**", "*.py")
    model_folder = os.path.join(root_folder, "models")
    model_pattern = os.path.join(model_folder, "**", "*.py")

    importlib.import_module("pythia.core.meter")

    files = glob.glob(tasks_pattern, recursive=True) + \
        glob.glob(model_pattern, recursive=True)

    for f in files:
        if f.endswith("task.py"):
            splits = f.split(os.sep)
            task_name = splits[-2]
            file_name = splits[-1]
            module_name = file_name[:file_name.find(".py")]
            importlib.import_module("pythia.tasks." + task_name + "."
                                    + module_name)
        elif f.find("models") != -1:
            splits = f.split(os.sep)
            file_name = splits[-1]
            module_name = file_name[:file_name.find(".py")]
            importlib.import_module("pythia.models." + module_name)
        elif f.endswith("builder.py"):
            splits = f.split(os.sep)
            dataset_name = splits[-2]
            file_name = splits[-1]
            module_name = file_name[:file_name.find(".py")]
            importlib.import_module("pythia.tasks.datasets." + dataset_name
                                    + "." + module_name)


def run():
    setup_imports()
    trainer = Trainer()
    trainer.load()
    trainer.train()


if __name__ == '__main__':
    run()
