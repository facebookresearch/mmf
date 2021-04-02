# Copyright (c) Facebook, Inc. and its affiliates.

import os
import unittest

from mmf.utils.general import get_mmf_root


def has_python_file(files):
    for f in files:
        if f.endswith(".py"):
            return True
    return False


def walk_and_assert_init(folder):
    for root, subfolders, files in os.walk(folder):
        if has_python_file(files):
            assert "__init__.py" in files, f"Folder {root} is missing __init__.py file"


def walk_and_assert_not_empty(folder):
    for root, subfolders, files in os.walk(folder):
        assert len(files) > 0 or len(subfolders) > 0, f"Folder {root} is empty"


class TestQualityChecks(unittest.TestCase):
    def _test_quality_check(self, fn):
        fn(get_mmf_root())
        fn(os.path.join(get_mmf_root(), "..", "mmf_cli"))
        fn(os.path.join(get_mmf_root(), "..", "tests"))

    def test_init_files_present(self):
        self._test_quality_check(walk_and_assert_init)

    def test_no_empty_folders(self):
        self._test_quality_check(walk_and_assert_not_empty)
