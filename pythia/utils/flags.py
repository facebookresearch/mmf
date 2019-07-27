# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import sys

from pythia.common.registry import registry


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()
        self.update_task_args()
        self.update_model_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        # TODO: Update default values
        self.parser.add_argument_group("Core Arguments")

        self.parser.add_argument(
            "--config", type=str, default=None, required=False, help="config yaml file"
        )

        self.parser.add_argument(
            "--tasks", type=str, required=True, help="Tasks for training"
        )
        self.parser.add_argument(
            "--datasets",
            type=str,
            required=False,
            default="all",
            help="Datasets to be used for required task",
        )
        self.parser.add_argument(
            "--model", type=str, required=True, help="Model for training"
        )
        self.parser.add_argument(
            "--run_type",
            type=str,
            default=None,
            help="Type of run. Default=train+predict",
        )
        self.parser.add_argument(
            "-exp",
            "--experiment_name",
            type=str,
            default=None,
            help="Name of the experiment",
        )

        self.parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="random seed, default None, meaning nothing will be seeded"
            " set seed to -1 if need a random seed"
            " between 1 and 100000",
        )
        self.parser.add_argument(
            "--config_overwrite",
            type=str,
            help="a json string to update yaml config file",
            default=None,
        )

        self.parser.add_argument(
            "--force_restart",
            action="store_true",
            help="flag to force clean previous result and restart training",
        )
        self.parser.add_argument(
            "--log_interval",
            type=int,
            default=None,
            help="Number of iterations after which we should log validation results",
        )
        self.parser.add_argument(
            "--snapshot_interval",
            type=int,
            default=None,
            help="Number of iterations after which  we should save snapshots",
        )
        self.parser.add_argument(
            "--max_iterations",
            type=int,
            default=None,
            help="Number of iterations after which  we should stop training",
        )
        self.parser.add_argument(
            "--max_epochs",
            type=int,
            default=None,
            help="Number of epochs after which "
            " we should stop training"
            " (mutually exclusive with max_iterations)",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=None,
            help="Batch size to be used for training "
            "If not passed it will default to config one",
        )
        self.parser.add_argument(
            "--save_dir",
            type=str,
            default="./save",
            help="Location for saving model checkpoint",
        )
        self.parser.add_argument(
            "--log_dir", type=str, default=None, help="Location for saving logs"
        )
        self.parser.add_argument(
            "--logger_level", type=str, default=None, help="Level of logging"
        )

        self.parser.add_argument(
            "--should_not_log",
            action="store_true",
            default=False,
            help="Set when you don't want logging to happen",
        )
        self.parser.add_argument(
            "-co",
            "--config_override",
            type=str,
            default=None,
            help="Use to override config from command line directly",
        )
        self.parser.add_argument(
            "--resume_file",
            type=str,
            default=None,
            help="File from which to resume checkpoint",
        )
        self.parser.add_argument(
            "--resume",
            type=bool,
            default=None,
            help="Use when you want to restore from automatic checkpoint",
        )
        self.parser.add_argument(
            "--evalai_inference",
            type=bool,
            default=None,
            help="Whether predictions should be made for EvalAI.",
        )
        self.parser.add_argument(
            "--verbose_dump",
            type=bool,
            default=None,
            help="Whether to do verbose dump of dataset"
            " samples, predictions and other things.",
        )
        self.parser.add_argument(
            "--lr_scheduler",
            type=bool,
            default=None,
            help="Use when you want to use lr scheduler",
        )
        self.parser.add_argument(
            "--clip_gradients",
            type=bool,
            default=None,
            help="Use when you want to clip gradients",
        )
        self.parser.add_argument(
            "--data_parallel",
            type=bool,
            default=None,
            help="Use when you want to use DataParallel",
        )
        self.parser.add_argument(
            "--distributed",
            type=bool,
            default=None,
            help="Use when you want to use DistributedDataParallel for training",
        )
        self.parser.add_argument(
            "-dev",
            "--device",
            type=str,
            default=None,
            help="Specify device to be used for training",
        )
        self.parser.add_argument(
            "-p", "--patience", type=int, default=None, help="Patience for early stop"
        )
        self.parser.add_argument(
            "-fr",
            "--fast_read",
            type=bool,
            default=None,
            help="If fast read should be activated",
        )
        self.parser.add_argument(
            "-pt",
            "--load_pretrained",
            type=int,
            default=None,
            help="If using a pretrained model. "
            "Must be used with --resume_file parameter "
            "to specify pretrained model checkpoint. "
            "Will load only specific layers if "
            "pretrained mapping is specified in config",
        )

        self.parser.add_argument(
            "-nw",
            "--num_workers",
            type=int,
            default=None,
            help="Number of workers for dataloaders",
        )
        self.parser.add_argument(
            "-lr",
            "--local_rank",
            type=int,
            default=None,
            help="Local rank of the current node",
        )
        self.parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="Modify config options from command line",
        )

    def update_task_args(self):
        args = sys.argv
        task_names = None
        for index, item in enumerate(args):
            if item == "--tasks":
                task_names = args[index + 1]

        if task_names is None:
            return

        task_names = map(lambda x: x.strip(), task_names.split(","))

        for task_name in task_names:
            task_class = registry.get_task_class(task_name)
            if task_class is None:
                return

            task_object = task_class()
            task_object.init_args(self.parser)

    def update_model_args(self):
        args = sys.argv
        model_name = None
        for index, item in enumerate(args):
            if item == "--model":
                model_name = args[index + 1]

        if model_name is None:
            return

        model_class = registry.get_model_class(model_name)
        if model_class is None:
            return

        model_class.init_args(self.parser)


flags = Flags()
