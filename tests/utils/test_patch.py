# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

from mmf.common.registry import registry
from mmf.utils.patch import restore_saved_modules, safecopy_modules


class TestClass:
    @staticmethod
    def test_function():
        return True


class TestUtilsPatch(unittest.TestCase):
    def test_safecopy_modules(self):

        safecopy_modules(["TestClass.test_function"], globals())
        original_functions = registry.get("original_patch_functions")
        self.assertTrue("TestClass.test_function" in original_functions)

        TestClass.test_function = lambda: False
        restore_saved_modules(globals())
        self.assertTrue(TestClass.test_function())
