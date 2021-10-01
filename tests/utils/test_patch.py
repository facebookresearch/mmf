# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

from mmf.utils.patch import restore_saved_modules, safecopy_modules


class TestClass:
    @staticmethod
    def test_function():
        return True


class TestUtilsPatch(unittest.TestCase):
    def test_safecopy_modules(self):
        from mmf.utils.patch import original_functions

        safecopy_modules(["TestClass.test_function"], globals())
        self.assertTrue("TestClass.test_function" in original_functions)

        TestClass.test_function = lambda: False
        restore_saved_modules(globals())
        self.assertTrue(TestClass.test_function())
