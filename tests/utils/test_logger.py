# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
import shutil
import tempfile
import unittest
from typing import Optional

from mmf.common.registry import registry
from mmf.utils.configuration import Configuration
from mmf.utils.file_io import PathManager
from mmf.utils.logger import Logger


class TestLogger(unittest.TestCase):

    _tmpdir: Optional[str] = None
    _tmpfile_write_contents: str = "print writer contents"

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.mkdtemp()
        args = argparse.Namespace()
        args.opts = [f"env.save_dir={cls._tmpdir}", f"model=cnn_lstm", f"dataset=clevr"]
        args.config_override = None
        configuration = Configuration(args)
        configuration.freeze()
        cls.config = configuration.get_config()
        registry.register("config", cls.config)
        cls.writer = Logger(cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)

    def test_logger_files(self) -> None:
        self.assertTrue(PathManager.exists(os.path.join(self._tmpdir, "train.log")))
        self.assertTrue(PathManager.exists(os.path.join(self._tmpdir, "logs")))

    def test_log_writer(self) -> None:
        self.writer.write(self._tmpfile_write_contents)
        f = PathManager.open(os.path.join(self._tmpdir, "train.log"))
        self.assertTrue(
            any(self._tmpfile_write_contents in line for line in f.readlines())
        )
