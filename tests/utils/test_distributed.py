# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import mmf.utils.distributed as distributed


class TestUtilsDistributed(unittest.TestCase):
    def test_object_byte_tensor_conversion(self):
        test_obj = [1, "2", {3: 4}, [5]]
        test_obj_bytes = distributed.object_to_byte_tensor(test_obj)
        test_obj_dec = distributed.byte_tensor_to_object(test_obj_bytes)
        self.assertEqual(test_obj_dec, test_obj)
