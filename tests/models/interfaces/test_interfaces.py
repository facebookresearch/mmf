# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import numpy as np

from mmf.models.mmbt import MMBT
from tests.test_utils import skip_if_no_network


class TestModelInterfaces(unittest.TestCase):
    @skip_if_no_network
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
