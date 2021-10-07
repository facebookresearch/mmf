# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
import transformers.models.vit.modeling_vit as vit
from mmf.modules.vit import ViTModel
from mmf.utils.patch import patch_vit
from tests.test_utils import setup_proxy
from torch import nn


patch_vit()


class TestViT(unittest.TestCase):
    def setUp(self):
        setup_proxy()

        config = {
            "layer_norm_eps": 0.0001,
            "hidden_size": 768,
            "num_hidden_layers": 2,
            "do_patch_embeddings": False,
            "add_pooling_layer": False,
            "return_dict": True,
        }
        hf_config = vit.ViTConfig()
        hf_config.update(config)
        self.model = ViTModel(hf_config)

    def test_model_init(self):
        self.assertTrue(isinstance(self.model, nn.Module))

    def test_model_forward(self):
        embeddings = torch.rand(32, 197, 768)
        output = self.model(embeddings, output_hidden_states=True)

        # import pdb; pdb.set_trace()

        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertEqual(output["last_hidden_state"].shape, torch.Size([32, 197, 768]))

        self.assertTrue(hasattr(output, "hidden_states"))
        self.assertEqual(len(output["hidden_states"]), 3)
