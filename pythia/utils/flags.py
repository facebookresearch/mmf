import argparse
import sys
import importlib

from pythia.constants import task_name_mapping


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_core_args()
        self.update_task_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Core Arguments")

        self.parser.add_argument("--config",
                                 type=str,
                                 required=False,
                                 help="config yaml file")
        self.parser.add_argument("--task",
                                 type=str,
                                 required=True,
                                 help="Task for training")
        self.parser.add_argument("--out_dir",
                                 type=str,
                                 default=None,
                                 help="output directory, default"
                                 " is current directory")
        self.parser.add_argument('--seed', type=int, default=1234,
                                 help="random seed, default 1234,"
                                 " set seed to -1 if need a random seed"
                                 " between 1 and 100000")
        self.parser.add_argument('--config_overwrite',
                                 type=str,
                                 help="a json string to update"
                                 " yaml config file",
                                 default=None)
        self.parser.add_argument("--force_restart", action='store_true',
                                 help="flag to force clean previous"
                                 " result and restart training")

    def update_task_args(self):
        args = sys.argv
        task_name = None
        for index, item in enumerate(args):
            if item == '--task':
                task_name = args[index + 1]

        if not task_name or task_name not in task_name_mapping:
            return

        task_module_name = "pythia.tasks"
        task_module = importlib.import_module(task_module_name)

        task_class = getattr(task_module, task_name)

        task_class.init_args(self.parser)


flags = Flags()
