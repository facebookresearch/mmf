# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import glob
import os
import shutil
import tempfile
import unittest
from typing import Optional

from mmf.common.registry import registry
from mmf.utils.configuration import Configuration
from mmf.utils.file_io import PathManager
from mmf.utils.logger import setup_logger, setup_output_folder


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
        setup_output_folder.cache_clear()
        setup_logger.cache_clear()
        cls.writer = setup_logger()

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        for handler in cls.writer.handlers:
            handler.close()
        if cls._tmpdir is not None:
            # set ignore_errors as Windows throw error due to Permission
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_logger_files(self) -> None:
        self.assertTrue(
            PathManager.exists(
                glob.glob(os.path.join(self._tmpdir, "logs", "train*"))[0]
            )
        )
        self.assertTrue(PathManager.exists(os.path.join(self._tmpdir, "train.log")))
        self.assertTrue(PathManager.exists(os.path.join(self._tmpdir, "logs")))

    def test_log_writer(self) -> None:
        self.writer.info(self._tmpfile_write_contents)
        f = PathManager.open(glob.glob(os.path.join(self._tmpdir, "logs", "train*"))[0])
        self.assertTrue(
            any(self._tmpfile_write_contents in line for line in f.readlines())
        )
        f = PathManager.open(os.path.join(self._tmpdir, "train.log"))
        self.assertTrue(
            any(self._tmpfile_write_contents in line for line in f.readlines())
        )
