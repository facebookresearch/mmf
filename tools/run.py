# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import importlib
import os
import random

import numpy as np
import torch

from pythia.common.registry import registry
from pythia.utils import distributed_utils
from pythia.utils.build_utils import build_trainer
from pythia.utils.configuration import Configuration
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
    datasets_folder = os.path.join(root_folder, "datasets")
    datasets_pattern = os.path.join(datasets_folder, "**", "*.py")
    model_folder = os.path.join(root_folder, "models")
    model_pattern = os.path.join(model_folder, "**", "*.py")

    importlib.import_module("pythia.common.meter")

    files = (
        glob.glob(datasets_pattern, recursive=True)
        + glob.glob(model_pattern, recursive=True)
        + glob.glob(trainer_pattern, recursive=True)
    )

    for f in files:
        if f.find("models") != -1:
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
            if task_name == "datasets" or dataset_name == "datasets":
                continue
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module(
                "pythia.datasets." + task_name + "." + dataset_name + "." + module_name
            )


def main(configuration, init_distributed=False):
    setup_imports()
    config = configuration.get_config()
    if torch.cuda.is_available():
        torch.cuda.set_device(config.device_id)
    if config.seed:
        if config.seed == -1:
            config.seed = random.randint(10000, 20000)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        # TODO: Reenable after project
        # random.seed(config.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
    if init_distributed:
        distributed_utils.distributed_init(config)
    trainer = build_trainer(configuration)
    trainer.load()
    trainer.train()


def distributed_main(device_id, configuration):
    config = configuration.get_config()
    config.device_id = device_id

    if config.distributed.rank is None:
        config.distributed.rank = config.start_rank + device_id

    main(configuration, init_distributed=True)


def run():
    setup_imports()
    parser = flags.get_parser()
    args = parser.parse_args()
    print(args)
    configuration = Configuration(args)
    # Do set runtime args which can be changed by pythia
    configuration.args = args
    config = configuration.get_config()
    config.start_rank = 0
    if config.distributed.init_method is None:
        distributed_utils.infer_init_method(config)

    if config.distributed.init_method is not None:
        if torch.cuda.device_count() > 1 and not config.distributed.no_spawn:
            config.start_rank = config.distributed.rank
            config.distributed.rank = None
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(configuration,),
                nprocs=torch.cuda.device_count(),
            )
        else:
            main(0, configuration)
    elif config.distributed.world_size > 1:
        assert config.distributed.world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        config.distributed.init_method = "tcp://localhost:{port}".format(port=port)
        config.distributed.rank = None
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(configuration,),
            nprocs=config.distributed.world_size,
        )
    else:
        config.device_id = 0
        main(configuration)
    # Log any errors that occur to log file
    # try:
    #     trainer.load()
    #     trainer.train()
    # except Exception as e:
    #     writer = getattr(trainer, "writer", None)

    #     if writer is not None:
    #         writer.write(e, "error", donot_print=True)
    #     if is_main_process():
    #         raise


if __name__ == "__main__":
    run()
