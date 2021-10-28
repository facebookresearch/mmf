# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import unittest
from unittest.mock import MagicMock

import mmf.modules.losses as losses
import torch
import torch.nn as nn
import torch.nn.functional as F
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


@registry.register_loss("mse_mae")
class TestMSEAndMAELoss(nn.Module):
    """Mean squared, absolute error loss.
    Calculates both losses and returns a dict with string keys.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        targets = sample_list["targets"]
        scores = model_output["scores"]
        loss = {"mse": F.mse_loss(scores, targets), "mae": F.l1_loss(scores, targets)}
        return loss


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

    def test_mmf_dict_loss(self):
        mse_mae_loss = losses.MMFLoss("mse_mae")
        torch.manual_seed(1234)
        random_tensor = torch.rand((1, 768))

        sample_list = SampleList()
        sample_list.dataset_type = "val"
        sample_list.dataset_name = "vqa2"
        sample_list["targets"] = random_tensor
        model_output = {"scores": random_tensor}

        output = mse_mae_loss(sample_list, model_output)

        self.assertEqual(output["val/vqa2/mse_mae/mse"].item(), 0.0)
        self.assertEqual(output["val/vqa2/mse_mae/mae"].item(), 0.0)

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

    def test_in_batch_hinge(self):
        in_batch_hinge_loss = losses.InBatchHinge(0.2, True)

        sample_list_input = dict()
        predicted = dict()

        # Test when the image and text have the same embeddings
        predicted["targets"] = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        predicted["scores"] = predicted["targets"]
        self.assertEqual(in_batch_hinge_loss(sample_list_input, predicted).item(), 0.0)

        # Test random initialized
        torch.manual_seed(1234)
        predicted["targets"] = torch.rand((5, 10))
        predicted["scores"] = torch.rand((5, 10))

        self.assertAlmostEqual(
            in_batch_hinge_loss(sample_list_input, predicted).item(), 6.5529985, 4
        )

    def test_mse_loss(self):
        mse_loss = losses.MSELoss()

        # Test random tensor but the same targets and scores
        torch.manual_seed(1234)
        random_tensor = torch.rand((1, 768))
        sample_list = {"targets": random_tensor}
        model_output = {"scores": random_tensor}
        self.assertEqual(mse_loss(sample_list, model_output).item(), 0.0)

    def test_cosine_embedding_loss(self):
        cos_emb_loss = losses.CosineEmbeddingLoss()

        # Test random tensor but the same targets and scores
        torch.manual_seed(1234)
        random_tensor = torch.rand((1, 768))
        sample_list = {"targets": random_tensor}
        model_output = {"scores": random_tensor}
        self.assertEqual(cos_emb_loss(sample_list, model_output).item(), 0.0)

    def test_bce_kl_loss(self):
        combined_loss = losses.BCEAndKLLoss(0.5)

        # Test random tensor but the same targets and scores
        torch.manual_seed(1234)
        random_tensor = torch.rand((1, 768))
        sample_list = {"targets": random_tensor}
        model_output = {"scores": random_tensor}
        loss_result = combined_loss(sample_list, model_output)
        self.assertAlmostEqual(loss_result["bce"].item(), 504.22253418, 4)
        self.assertAlmostEqual(loss_result["kl"].item(), 0.031847, 4)

    def test_refiner_ms_loss(self):
        refiner_ms_loss = losses.RefinerMSLoss(
            alpha=50, beta=2, base=0.5, margin=0.1, epsilon=1e-16
        )

        torch.manual_seed(1234)
        random_tensor = torch.rand((1, 768))
        sample_list = {"targets": random_tensor}
        model_output = {"scores": random_tensor}

        loss_result = refiner_ms_loss(sample_list, model_output)
        self.assertEqual(loss_result, 0.0)

    def test_ms_loss(self):
        ms_loss = losses.MSLoss(
            alpha=50, beta=2, margin=0.5, hard_mining=True, is_multilabel=False
        )

        torch.manual_seed(1234)

        label_tensor = torch.Tensor([0, 0, 0, 0, 0])
        fused_tensor = torch.randn(5, 768)
        sample_list = {"targets": label_tensor}
        model_output = {"fused_embedding": fused_tensor}
        loss_result = ms_loss(sample_list, model_output)
        self.assertEqual(loss_result, 0.0)

        label_tensor = torch.Tensor([1, 1, 1, 1, 1])
        loss_result = ms_loss(sample_list, model_output)
        self.assertEqual(loss_result, 0.0)

        label_tensor = torch.Tensor([1, 1, 1, 1, 1])
        loss_result = ms_loss(sample_list, model_output)
        self.assertEqual(loss_result, 0.0)

    def test_refiner_contrastive_loss(self):
        refiner_contrastive_loss = losses.RefinerContrastiveLoss(
            sim_thresh=0.1, epsilon=1e-16
        )

        inputs = torch.rand((10, 768))
        targets = inputs

        sample_list = {"targets": targets}
        model_output = {"scores": inputs}

        loss_result = refiner_contrastive_loss(sample_list, model_output)
        self.assertEqual(loss_result, 0.0)
