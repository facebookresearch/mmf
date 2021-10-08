# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.modules.vit import ViTModel
from tests.test_utils import setup_proxy, skip_if_old_transformers
from torch import nn


@skip_if_old_transformers
class TestViT(unittest.TestCase):
    def setUp(self):
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
        config = vit.ViTConfig(**config)
        self.model = ViTModel(config)

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
