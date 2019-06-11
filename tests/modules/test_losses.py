# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import pythia.modules.losses as losses
import torch


class TestModuleLosses(unittest.TestCase):
    def test_caption_cross_entropy(self):
        caption_ce_loss = losses.CaptionCrossEntropyLoss()

        expected = dict()
        predicted = dict()

        # Test complete match
        expected["targets"] = torch.empty((1, 10), dtype=torch.long)
        expected["targets"].fill_(4)
        predicted["scores"] = torch.zeros((1, 10, 10))
        predicted["scores"][:, :, 4] = 100.0

        self.assertEqual(caption_ce_loss(expected, predicted).item(), 0.0)

        # Test random initialized
        torch.manual_seed(1234)
        expected["targets"] = torch.randint(0, 9491, (5, 10))
        predicted["scores"] = torch.rand((5, 10, 9491))

        self.assertAlmostEqual(caption_ce_loss(expected, predicted).item(), 9.2507, 4)
