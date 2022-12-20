# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

from mmf.common.registry import registry
from mmf.utils.patch import (
    ORIGINAL_PATCH_FUNCTIONS_KEY,
    restore_saved_modules,
    safecopy_modules,
)


class TestClass:
    @staticmethod
    def test_function():
        return True


class TestUtilsPatch(unittest.TestCase):
    def setUp(self):
        registry.register(ORIGINAL_PATCH_FUNCTIONS_KEY, {})

    def test_safecopy_modules(self):
        safecopy_modules(["TestClass.test_function"], {"TestClass": TestClass})
        original_functions = registry.get(ORIGINAL_PATCH_FUNCTIONS_KEY)
        self.assertTrue("TestClass.test_function" in original_functions)

        TestClass.test_function = lambda: False
        restore_saved_modules({"TestClass": TestClass})
        self.assertTrue(TestClass.test_function())
