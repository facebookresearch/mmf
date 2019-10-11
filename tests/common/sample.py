# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from pythia.common.sample import Sample


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
