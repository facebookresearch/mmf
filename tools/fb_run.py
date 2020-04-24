# Copyright (c) Facebook, Inc. and its affiliates.

import torch.fb.rendezvous.zeus  # noqa
from fvcore.fb.manifold import ManifoldPathHandler  # noqa

from mmf.utils.file_io import PathManager
from mmf.utils.flags import flags
from tools.run import distributed_main, main


def get_fb_training_parser():
    parser = flags.get_parser()
    # [FB] Additional FB specific cmd args
    return parser


def fb_run(device_id, configuration, start_rank, log_path=None):
    """[FB] entry point for each worker process."""
    config = configuration.get_config()
    config.distributed.rank = start_rank + device_id

    # support Manifold for checkpoints
    PathManager.register_handler(ManifoldPathHandler(max_parallel=16, timeout_sec=1800))

    if config.distributed.world_size > 1:
        distributed_main(device_id, configuration)
    else:
        main(configuration)
