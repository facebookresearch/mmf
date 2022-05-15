# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.sample import (
    convert_batch_to_sample_list,
    Sample,
    SampleList,
    to_device,
)


class TestSample(unittest.TestCase):
    def test_sample_working(self):
        initial = Sample()
        initial.x = 1
        initial["y"] = 2
        # Assert setter and getter
        self.assertEqual(initial.x, 1)
        self.assertEqual(initial["x"], 1)
        self.assertEqual(initial.y, 2)
        self.assertEqual(initial["y"], 2)

        update_dict = {"a": 3, "b": {"c": 4}}

        initial.update(update_dict)
        self.assertEqual(initial.a, 3)
        self.assertEqual(initial["a"], 3)
        self.assertEqual(initial.b.c, 4)
        self.assertEqual(initial["b"].c, 4)


class TestSampleList(unittest.TestCase):
    @test_utils.skip_if_no_cuda
    def test_pin_memory(self):
        sample_list = test_utils.build_random_sample_list()
        sample_list.pin_memory()

        pin_list = [sample_list.y, sample_list.z.y]
        non_pin_list = [sample_list.x, sample_list.z.x]

        all_pinned = True

        for pin in pin_list:
            all_pinned = all_pinned and pin.is_pinned()

        self.assertTrue(all_pinned)

        any_pinned = False

        for pin in non_pin_list:
            any_pinned = any_pinned or (hasattr(pin, "is_pinned") and pin.is_pinned())

        self.assertFalse(any_pinned)

    def test_to_dict(self):
        sample_list = test_utils.build_random_sample_list()
        sample_dict = sample_list.to_dict()
        self.assertTrue(isinstance(sample_dict, dict))
        # hasattr won't work anymore
        self.assertFalse(hasattr(sample_dict, "x"))
        keys_to_assert = ["x", "y", "z", "z.x", "z.y"]

        all_keys = True
        for key in keys_to_assert:
            current = sample_dict
            if "." in key:
                sub_keys = key.split(".")
                for sub_key in sub_keys:
                    all_keys = all_keys and sub_key in current
                    current = current[sub_key]
            else:
                all_keys = all_keys and key in current

        self.assertTrue(all_keys)
        self.assertTrue(isinstance(sample_dict, dict))


class TestFunctions(unittest.TestCase):
    def test_to_device(self):
        sample_list = test_utils.build_random_sample_list()

        modified = to_device(sample_list, "cpu")
        self.assertEqual(modified.get_device(), torch.device("cpu"))

        modified = to_device(sample_list, torch.device("cpu"))
        self.assertEqual(modified.get_device(), torch.device("cpu"))

        modified = to_device(sample_list, "cuda")

        if torch.cuda.is_available():
            self.assertEqual(modified.get_device(), torch.device("cuda:0"))
        else:
            self.assertEqual(modified.get_device(), torch.device("cpu"))

        double_modified = to_device(modified, modified.get_device())
        self.assertTrue(double_modified is modified)

        custom_batch = [{"a": 1}]
        self.assertEqual(to_device(custom_batch), custom_batch)

    def test_convert_batch_to_sample_list(self):
        # Test list conversion
        batch = [{"a": torch.tensor([1.0, 1.0])}, {"a": torch.tensor([2.0, 2.0])}]
        sample_list = convert_batch_to_sample_list(batch)
        expected_a = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        self.assertTrue(torch.equal(expected_a, sample_list.a))

        # Test single element list, samplelist
        sample_list = SampleList()
        sample_list.add_field("a", expected_a)
        parsed_sample_list = convert_batch_to_sample_list([sample_list])
        self.assertTrue(isinstance(parsed_sample_list, SampleList))
        self.assertTrue("a" in parsed_sample_list)
        self.assertTrue(torch.equal(expected_a, parsed_sample_list.a))

        # Test no tensor field
        batch = [{"a": [1]}, {"a": [2]}]
        sample_list = convert_batch_to_sample_list(batch)
        self.assertTrue(sample_list.a, [[1], [2]])
