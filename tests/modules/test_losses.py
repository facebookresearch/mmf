# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import unittest
from unittest.mock import MagicMock

import mmf.modules.losses as losses
import torch
from mmf.common.registry import registry
from mmf.common.sample import SampleList


RETURN_VALUE = torch.tensor(1.0)


def build_loss_side_effect(return_value=RETURN_VALUE):
    def loss_side_effect(item):
        loss_object_mock = MagicMock(return_value=return_value)
        loss_class_mock = MagicMock(return_value=loss_object_mock)
        valid_losses = ["cross_entropy", "multi"]
        if isinstance(item, collections.abc.MutableMapping):
            if item["type"] not in valid_losses:
                return None
        elif item not in valid_losses:
            return None
        else:
            return loss_class_mock

    return loss_side_effect


class TestModuleLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.registry_loss_class = registry.get_loss_class

    def tearDown(self):
        registry.get_loss_class = self.registry_loss_class

    def test_mmf_loss(self):
        get_loss_class_mock = MagicMock(side_effect=build_loss_side_effect())
        registry.get_loss_class = get_loss_class_mock
        # Test if MMFLoss accepts empty parameters
        self.assertRaises(ValueError, losses.MMFLoss)
        self.assertTrue(losses.MMFLoss({"type": "cross_entropy"}).name, "cross_entropy")
        self.assertTrue(losses.MMFLoss("cross_entropy").name, "cross_entropy")
        self.assertRaises(AssertionError, losses.MMFLoss, [])
        # Multi requires dict
        self.assertRaises(AssertionError, losses.MMFLoss, "multi")

        cross_entropy = losses.MMFLoss("cross_entropy")
        cross_entropy_from_dict = losses.MMFLoss({"type": "cross_entropy"})
        sample_list = SampleList()
        sample_list.dataset_type = "val"
        sample_list.dataset_name = "vqa2"

        output = cross_entropy(sample_list, {})
        output_from_dict = cross_entropy_from_dict(sample_list, {})

        self.assertEqual(output, {"val/vqa2/cross_entropy": torch.tensor(1.0)})
        self.assertEqual(output_from_dict, output)

        get_loss_class_mock.side_effect = build_loss_side_effect(1.0)
        output = cross_entropy(sample_list, {})

        self.assertEqual(output, {"val/vqa2/cross_entropy": torch.tensor(1.0)})
        self.assertEqual(output_from_dict, output)

        self.assertTrue(get_loss_class_mock.called)
        self.assertEqual(get_loss_class_mock.call_count, 5)

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
