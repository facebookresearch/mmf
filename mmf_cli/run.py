#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import random
import typing

import torch

from mmf.common.registry import registry
from mmf.utils.build import build_config, build_trainer
from mmf.utils.configuration import Configuration
from mmf.utils.distributed import distributed_init, infer_init_method
from mmf.utils.env import set_seed, setup_imports
from mmf.utils.flags import flags
from mmf.utils.logger import Logger


def main(configuration, init_distributed=False, predict=False):
    # A reload might be needed for imports
    setup_imports()
    configuration.import_user_dir()
    config = configuration.get_config()

    if torch.cuda.is_available():
        torch.cuda.set_device(config.device_id)
        torch.cuda.init()

    if init_distributed:
        distributed_init(config)

    config.training.seed = set_seed(config.training.seed)
    registry.register("seed", config.training.seed)
    print(f"Using seed {config.training.seed}")

    config = build_config(configuration)

    # Logger should be registered after config is registered
    registry.register("writer", Logger(config, name="mmf.train"))
    trainer = build_trainer(config)
    trainer.load()
    if predict:
        trainer.inference()
    else:
        trainer.train()


def distributed_main(device_id, configuration, predict=False):
    config = configuration.get_config()
    config.device_id = device_id

    if config.distributed.rank is None:
        config.distributed.rank = config.start_rank + device_id

    main(configuration, init_distributed=True, predict=predict)


def run(opts: typing.Optional[typing.List[str]] = None, predict: bool = False):
    """Run starts a job based on the command passed from the command line.
    You can optionally run the mmf job programmatically by passing an optlist as opts.

    Args:
        opts (typing.Optional[typing.List[str]], optional): Optlist which can be used.
            to override opts programmatically. For e.g. if you pass
            opts = ["training.batch_size=64", "checkpoint.resume=True"], this will
            set the batch size to 64 and resume from the checkpoint if present.
            Defaults to None.
        predict (bool, optional): If predict is passed True, then the program runs in
            prediction mode. Defaults to False.
    """
    setup_imports()

    if opts is None:
        parser = flags.get_parser()
        args = parser.parse_args()
    else:
        args = argparse.Namespace(config_override=None)
        args.opts = opts

    print(args)
    configuration = Configuration(args)
    # Do set runtime args which can be changed by MMF
    configuration.args = args
    config = configuration.get_config()
    config.start_rank = 0
    if config.distributed.init_method is None:
        infer_init_method(config)

    if config.distributed.init_method is not None:
        if torch.cuda.device_count() > 1 and not config.distributed.no_spawn:
            config.start_rank = config.distributed.rank
            config.distributed.rank = None
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(configuration, predict),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(0, configuration, predict)
    elif config.distributed.world_size > 1:
        assert config.distributed.world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        config.distributed.init_method = f"tcp://localhost:{port}"
        config.distributed.rank = None
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(configuration, predict),
            nprocs=config.distributed.world_size,
        )
    else:
        config.device_id = 0
        main(configuration, predict=predict)


if __name__ == "__main__":
    run()
