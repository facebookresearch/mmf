# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import numpy as np

import tests.test_utils as test_utils
from mmf.models.mmbt import MMBT


class TestModelInterfaces(unittest.TestCase):
    @test_utils.skip_if_no_network
    @test_utils.skip_if_windows
    @test_utils.skip_if_macos
    def test_mmbt_hm_interface(self):
        model = MMBT.from_pretrained("mmbt.hateful_memes.images")
        result = model.classify(
            "https://i.imgur.com/tEcsk5q.jpg", "look how many people love you"
        )

        self.assertEqual(result["label"], 0)
        np.testing.assert_almost_equal(result["confidence"], 0.9999, decimal=4)
        result = model.classify(
            "https://i.imgur.com/tEcsk5q.jpg", "they have the privilege"
        )
        self.assertEqual(result["label"], 0)
        np.testing.assert_almost_equal(result["confidence"], 0.9869, decimal=4)
        result = model.classify(
            "https://i.imgur.com/tEcsk5q.jpg", "a black man doing nothing"
        )
        self.assertEqual(result["label"], 1)
        np.testing.assert_almost_equal(result["confidence"], 0.6941, decimal=4)
