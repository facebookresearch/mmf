# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.common.sample import Sample
from mmf.models.transformers.heads.itm import ITM
from mmf.models.transformers.heads.mlm import MLM
from mmf.models.transformers.heads.mlp import MLP
from mmf.models.transformers.heads.refiner import Refiner
from mmf.models.transformers.heads.refnet_classifier import RefinerClassifier
from omegaconf import OmegaConf
from tests.test_utils import skip_if_no_cuda


class TestMLMHead(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {"type": "mlm", "freeze": False, "vocab_size": 1000, "hidden_size": 768}
        )

    @skip_if_no_cuda
    def test_forward(self):
        module = MLM(self.config).to("cuda")
        sequence_input = torch.rand(size=(1, 64, 768), dtype=torch.float, device="cuda")
        encoder_output = [sequence_input, sequence_input]
        processed_sample_list = Sample()
        processed_sample_list["mlm_labels"] = {}
        processed_sample_list["mlm_labels"]["combined_labels"] = torch.ones(
            size=(1, 64), dtype=torch.long, device="cuda"
        )

        output = module(sequence_input, encoder_output, processed_sample_list)

        self.assertTrue("logits" in output)
        self.assertTrue("losses" in output and "masked_lm_loss" in output["losses"])

        self.assertEqual(output["logits"].shape, torch.Size([64, 1000]))


class TestMLPHead(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {"type": "mlp", "num_labels": 2, "hidden_size": 768}
        )

    def test_forward(self):
        module = MLP(self.config)
        sequence_input = torch.ones(size=(1, 64, 768), dtype=torch.float)
        encoder_output = [sequence_input, sequence_input]
        processed_sample_list = {}

        output = module(sequence_input, encoder_output, processed_sample_list)

        self.assertTrue("scores" in output)
        self.assertEqual(output["scores"].shape, torch.Size([1, 2]))


class TestITMHead(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create({"type": "itm", "hidden_size": 768})

    def test_forward(self):
        module = ITM(self.config)
        sequence_input = torch.ones(size=(1, 64, 768), dtype=torch.float)
        encoder_output = [sequence_input, sequence_input]
        processed_sample_list = Sample()
        processed_sample_list["itm_labels"] = {}
        processed_sample_list["itm_labels"]["is_correct"] = torch.tensor(
            False, dtype=torch.long
        )

        output = module(sequence_input, encoder_output, processed_sample_list)

        self.assertTrue("itm_loss" in output["losses"])
        self.assertEqual(output["losses"]["itm_loss"].shape, torch.Size([]))


class TestMutilayerMLPHead(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {
                "type": "mlp",
                "num_labels": 2,
                "hidden_size": 768,
                "num_layers": 2,
                "in_dim": 768,
                "pooler_name": "bert_pooler",
            }
        )

    def test_forward(self):
        module = MLP(self.config)
        sequence_input = torch.ones(size=(1, 64, 768), dtype=torch.float)
        encoder_output = [sequence_input, sequence_input]
        processed_sample_list = {}

        output = module(sequence_input, encoder_output, processed_sample_list)

        self.assertTrue("scores" in output)
        self.assertEqual(output["scores"].shape, torch.Size([1, 2]))


class TestRefinerHead(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {
                "type": "refiner",
                "refiner_target_pooler": "average_k_from_last",
                "refiner_target_layer_depth": 1,
            }
        )

    def test_forward(self):
        module = Refiner(self.config)
        sequence_input = torch.ones(size=(1, 128, 768), dtype=torch.float)
        encoder_output = [sequence_input, sequence_input]
        processed_sample_list = {}
        processed_sample_list["masks"] = {}
        processed_sample_list["masks"]["text"] = torch.ones(
            size=(1, 64), dtype=torch.long
        )
        processed_sample_list["masks"]["image"] = torch.ones(
            size=(1, 64), dtype=torch.long
        )
        output = module(sequence_input, encoder_output, processed_sample_list)
        self.assertTrue("losses" in output)
        self.assertTrue("fused_embedding" in output)
        self.assertTrue("refiner_ss_loss" in output["losses"].keys())
        self.assertEqual(output["fused_embedding"].shape, torch.Size([1, 768]))


class TestRefNetClassifierHead(unittest.TestCase):
    def setUp(self):

        self.refiner_config = {
            "type": "refiner",
            "refiner_target_pooler": "average_k_from_last",
            "refiner_target_layer_depth": 1,
        }

        self.mlp_loss_config = OmegaConf.create(
            {
                "config": {"type": "mlp"},
                "loss_name": "classification_loss",
                "loss": "cross_entropy",
                "max_sample_size": 10000,
            }
        )

        self.config = OmegaConf.create(
            {
                "type": "refiner_classifier",
                "use_msloss": True,
                "refiner_config": self.refiner_config,
                "mlp_loss_config": self.mlp_loss_config,
            }
        )

    def test_forward(self):
        module = RefinerClassifier(self.config)
        sequence_input = torch.ones(size=(5, 128, 768), dtype=torch.float)
        encoder_output = [sequence_input, sequence_input]
        processed_sample_list = {}
        processed_sample_list["masks"] = {}
        processed_sample_list["masks"]["text"] = torch.ones(
            size=(5, 64), dtype=torch.long
        )
        processed_sample_list["masks"]["image"] = torch.ones(
            size=(5, 64), dtype=torch.long
        )
        processed_sample_list["target_key"] = {}
        processed_sample_list["target_key"]["targets"] = torch.empty(
            5, dtype=torch.long
        ).random_(2)
        output = module(sequence_input, encoder_output, processed_sample_list)
        self.assertTrue("losses" in output)
        self.assertTrue("fused_embedding" in output)
        self.assertTrue("ms_loss" in output["losses"].keys())
