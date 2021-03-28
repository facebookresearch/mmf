# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.common.sample import Sample
from mmf.models.transformers.heads.mlm import MLM
from mmf.models.transformers.heads.mlp import MLP
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
