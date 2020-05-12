#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
import random

import torch

from mmf.common.registry import registry
from mmf.utils.build import build_trainer
from mmf.utils.configuration import Configuration
from mmf.utils.distributed import distributed_init, infer_init_method
from mmf.utils.env import set_seed
from mmf.utils.flags import flags
from mmf.utils.general import setup_imports
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
    print("Using seed {}".format(config.training.seed))

    registry.register("writer", Logger(config, name="mmf.train"))

    trainer = build_trainer(configuration)
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


def run(predict=False):
    setup_imports()
    parser = flags.get_parser()
    args = parser.parse_args()
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
        config.distributed.init_method = "tcp://localhost:{port}".format(port=port)
        config.distributed.rank = None
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(configuration, predict),
            nprocs=config.distributed.world_size,
        )
    else:
        config.device_id = 0
        main(configuration, predict)


if __name__ == "__main__":
    run()
