# Copyright (c) Facebook, Inc. and its affiliates.

import os
import shutil
import tempfile
import unittest
import uuid
from typing import Optional

from mmf.utils.file_io import PathManager


class TestFileIO(unittest.TestCase):

    _tmpdir: Optional[str] = None
    _tmpfile: Optional[str] = None
    _tmpfile_contents = "Hello, World"

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.mkdtemp()
        with open(os.path.join(cls._tmpdir, "test.txt"), "w") as f:
            cls._tmpfile = f.name
            f.write(cls._tmpfile_contents)
            f.flush()

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)

    def test_file_io_open(self):
        with PathManager.open(self._tmpfile, mode="r") as f:
            s = f.read()
        self.assertEqual(s, self._tmpfile_contents)

    def test_file_io_copy(self):
        PathManager.copy(self._tmpfile, os.path.join(self._tmpdir, "test_copy.txt"))
        with open(os.path.join(self._tmpdir, "test_copy.txt")) as f:
            s = f.read()
        self.assertEqual(s, self._tmpfile_contents)

    def test_file_io_exists(self):
        self.assertEqual(
            PathManager.exists(self._tmpfile), os.path.exists(self._tmpfile)
        )
        fake_path = os.path.join(self._tmpdir, uuid.uuid4().hex)
        self.assertEqual(PathManager.exists(fake_path), os.path.exists(fake_path))

    def test_file_io_mkdirs(self):
        dir_path = os.path.join(self._tmpdir, "test_dir")
        PathManager.mkdirs(dir_path)
        self.assertTrue(os.path.isdir(dir_path))
