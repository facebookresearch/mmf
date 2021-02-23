# Copyright (c) Facebook, Inc. and its affiliates.

from iopath.common.file_io import PathManager as pm


PathManager = pm()

try:
    # [FB only] register internal file IO handlers
    from mmf.utils.fb.file_io_handlers import register_handlers

    register_handlers(PathManager)
except ImportError:
    pass
