import argparse
import sys
import importlib

from pythia.core.constants import task_name_mapping


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()
        self.update_task_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        # TODO: Update default values
        self.parser.add_argument_group("Core Arguments")

        self.parser.add_argument("--config",
                                 type=str,
                                 default=None,
                                 required=False,
                                 help="config yaml file")

        self.parser.add_argument("--task",
                                 type=str,
                                 required=True,
                                 help="Task for training")
        self.parser.add_argument("--model",
                                 type=str,
                                 required=True,
                                 help="Model for training")

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
        self.parser.add_argument("--log_interval", type=int, default=None,
                                 help="Number of iterations after which"
                                 " we should log validation results")
        self.parser.add_argument("--snapshot_interval", type=int, default=None,
                                 help="Number of iterations after which "
                                 " we should save snapshots")
        self.parser.add_argument("--max_iterations", type=int, default=None,
                                 help="Number of iterations after which "
                                 " we should stop training")
        self.parser.add_argument("--max_epochs", type=int, default=None,
                                 help="Number of epochs after which "
                                 " we should stop training"
                                 " (mutually exclusive with max_iterations)")
        self.parser.add_argument("--batch_size", type=int, default=None,
                                 help="Batch size to be used for training "
                                 "If not passed it will default to config one")
        self.parser.add_argument("--save_loc", type=str, default="./save",
                                 help="Location for saving model checkpoint")
        self.parser.add_argument("--log_dir", type=str, default="./logs",
                                 help="Location for saving logs")
        self.parser.add_argument("--should_not_log", action="store_true",
                                 default=False, help="Set when you don't want"
                                 " logging to happen")

        self.parser.add_argument("--resume_file", type=str, default=None,
                                 help="File from which to resume checkpoint")
        self.parser.add_argument("--resume", type=bool,
                                 default=None,
                                 help="Use when you want to restore from "
                                 "automatic checkpoint")
        self.parser.add_argument("--lr_scheduler", type=bool,
                                 default=None,
                                 help="Use when you want to use lr scheduler")
        self.parser.add_argument("--clip_gradients", type=bool,
                                 default=None,
                                 help="Use when you want to clip gradients")
        self.parser.add_argument("-cpu", "--no_cuda", action="store_true",
                                 default=False, help="Use when you don't want"
                                 " to use CUDA")

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

        task_class_name = task_name_mapping[task_name]
        task_class = getattr(task_module, task_class_name)

        task_object = task_class('test')
        task_object.init_args(self.parser)


flags = Flags()
