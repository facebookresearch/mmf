# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.common.sample import Sample
from mmf.models.transformers.heads.itm import ITM
from mmf.models.transformers.heads.mlm import MLM
from mmf.models.transformers.heads.mlp import MLP
from mmf.models.transformers.heads.mrc import MRC
from mmf.models.transformers.heads.mrfr import MRFR
from mmf.models.transformers.heads.refiner import Refiner
from mmf.models.transformers.heads.refnet_classifier import RefinerClassifier
from mmf.models.transformers.heads.wra import WRA
from omegaconf import OmegaConf
from tests.test_utils import skip_if_no_cuda
from torch import nn


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


class TestMRCHead(unittest.TestCase):
    def setUp(self):
        bs = 8
        num_feat = 64
        feat_dim = 768
        label_dim = 100
        self.sequence_input = torch.ones(
            size=(bs, num_feat, feat_dim), dtype=torch.float
        )
        self.processed_sample_list = Sample()
        label_targets = torch.rand((bs, num_feat, label_dim))
        self.processed_sample_list["region_class"] = label_targets.contiguous().view(
            -1, label_dim
        )
        self.processed_sample_list["image_region_mask"] = torch.ones(
            (bs, num_feat)
        ).bool()

    def test_forward_kldiv(self):
        config = OmegaConf.create({"hidden_size": 768, "label_dim": 100})
        module = MRC(**config)
        output = module(self.sequence_input, self.processed_sample_list)
        self.assertTrue("mrc_loss" in output["losses"])
        self.assertEqual(output["losses"]["mrc_loss"].shape, torch.Size([]))

    def test_forward_ce(self):
        config = OmegaConf.create(
            {"use_kl": False, "hidden_size": 768, "label_dim": 100}
        )
        module = MRC(**config)
        output = module(self.sequence_input, self.processed_sample_list)
        self.assertTrue("mrc_loss" in output["losses"])
        self.assertEqual(output["losses"]["mrc_loss"].shape, torch.Size([]))


class TestMRFRHead(unittest.TestCase):
    def setUp(self):
        bs = 8
        num_feat = 64
        feat_dim = 768
        img_dim = 1024  # feature proj output dim
        self.sequence_input = torch.ones(
            size=(bs, num_feat, feat_dim), dtype=torch.float
        )
        self.processed_sample_list = Sample()
        feat_targets = torch.zeros((bs, num_feat, img_dim))
        self.processed_sample_list[
            "mrfr_region_target"
        ] = feat_targets.contiguous().view(-1, img_dim)
        self.processed_sample_list["mrfr_region_mask"] = torch.ones(
            (bs, num_feat)
        ).bool()

        self.img_embedding_weight = nn.Parameter(torch.rand((feat_dim, img_dim)))

    def test_forward(self):
        config = OmegaConf.create({"hidden_size": 768, "img_dim": 1024})
        module = MRFR(self.img_embedding_weight, **config)
        output = module(self.sequence_input, self.processed_sample_list)
        self.assertTrue("mrfr_loss" in output["losses"])
        self.assertEqual(output["losses"]["mrfr_loss"].shape, torch.Size([]))

    def test_linear_proj_param_is_shared(self):
        config = OmegaConf.create({"hidden_size": 768, "img_dim": 1024})
        module = MRFR(self.img_embedding_weight, **config)
        with torch.no_grad():
            self.img_embedding_weight *= 0
            output = module(self.sequence_input, self.processed_sample_list)

        self.assertTrue(
            torch.equal(module.linear_proj_weight, self.img_embedding_weight)
        )
        self.assertEqual(output["losses"]["mrfr_loss"], 0)


class TestWRAHead(unittest.TestCase):
    def setUp(self):
        bs = 8
        num_feat = 64
        feat_dim = 768
        img_dim = 1024  # feature proj output dim
        sentence_len = 25
        num_img_feat = num_feat - sentence_len
        self.sequence_input = torch.ones(
            size=(bs, num_feat, feat_dim), dtype=torch.float
        )
        input_ids = torch.ones((bs, sentence_len))
        img_feat = torch.rand((bs, num_img_feat, img_dim))
        txt_pad = torch.zeros((bs, sentence_len), dtype=torch.long)
        img_pad = torch.zeros((bs, num_img_feat), dtype=torch.long)
        ot_inputs = {"txt_pad": txt_pad, "img_pad": img_pad}
        is_correct = torch.randint(2, (bs,))

        self.processed_sample_list = Sample()
        self.processed_sample_list["input_ids"] = input_ids
        self.processed_sample_list["image_feat"] = img_feat
        self.processed_sample_list["wra_info"] = ot_inputs
        self.processed_sample_list["is_correct"] = is_correct

    def test_forward(self):
        module = WRA()
        output = module(self.sequence_input, self.processed_sample_list)
        self.assertTrue("wra_loss" in output["losses"])
        self.assertEqual(output["losses"]["wra_loss"].shape, torch.Size([]))
