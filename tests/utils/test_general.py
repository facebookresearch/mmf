# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from mmf.utils.general import dict_to_string, get_overlap_score


class TestUtilsGeneral(unittest.TestCase):
    def test_dict_to_string(self):
        dictionary = {"one": 1, "two": 2, "three": 3}
        expected = "one: 1.0000, two: 2.0000, three: 3.0000"

        self.assertEqual(dict_to_string(dictionary), expected)

    # TODO: Move later to configuration tests
    # def test_nested_dict_update(self):
    #     # Updates value
    #     dictionary = {"level1": {"level2": {"levelA": 0, "levelB": 1}}}
    #     update = {"level1": {"level2": {"levelB": 10}}}
    #     expected = {"level1": {"level2": {"levelA": 0, "levelB": 10}}}
    #
    #     self.assertEqual(nested_dict_update(dictionary, update), expected)
    #
    #     # Adds new value
    #     dictionary = {"level1": {"level2": {"levelA": 0}}}
    #     update = {"level1": {"level2": {"levelB": 10}}}
    #     expected = {"level1": {"level2": {"levelA": 0, "levelB": 10}}}
    #
    #     self.assertEqual(nested_dict_update(dictionary, update), expected)

    def test_get_overlap_score(self):
        # Full overlap
        candidate = "pythia"
        target = "pythia"
        self.assertEqual(get_overlap_score(candidate, target), 1.0)

        # Partial overlap
        candidate = "pythia"
        target = "python"
        self.assertEqual(get_overlap_score(candidate, target), 2 / 3)

        # No overlap
        candidate = "pythia"
        target = "vqa"
        self.assertEqual(get_overlap_score(candidate, target), 0.0)
