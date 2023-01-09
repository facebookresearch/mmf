# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.modules.vit import ViTModel
from omegaconf import OmegaConf
from tests.test_utils import setup_proxy, skip_if_old_transformers
from torch import nn


@skip_if_old_transformers(min_version="4.5.0")
class TestViT(unittest.TestCase):
    def setUp(self):
        try:
            import transformers3.models.vit.modeling_vit as vit
        except ImportError:
            import transformers.models.vit.modeling_vit as vit

        setup_proxy()
        config = {
            "layer_norm_eps": 0.0001,
            "hidden_size": 768,
            "num_hidden_layers": 2,
            "do_patch_embeddings": False,
            "add_pooling_layer": False,
            "return_dict": True,
        }
        hf_config = vit.ViTConfig(**config)
        self.model = ViTModel(hf_config)

    def test_model_static_constructor_from_config(self):
        config = OmegaConf.create(
            {
                "pretrained_model_name": "google/vit-base-patch16-224",
                "do_patch_embeddings": False,
                "add_pooling_layer": False,
                "return_dict": True,
            }
        )
        pretrained_model, _ = ViTModel.from_config(config)
        embeddings = torch.rand(32, 197, 768)
        output = pretrained_model(embeddings, output_hidden_states=False)

        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertEqual(output["last_hidden_state"].shape, torch.Size([32, 197, 768]))

    def test_model_init(self):
        self.assertTrue(isinstance(self.model, nn.Module))

    def test_model_forward(self):
        embeddings = torch.rand(32, 197, 768)
        output = self.model(embeddings, output_hidden_states=True)

        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertEqual(output["last_hidden_state"].shape, torch.Size([32, 197, 768]))

        self.assertTrue(hasattr(output, "hidden_states"))
        self.assertEqual(len(output["hidden_states"]), 3)
