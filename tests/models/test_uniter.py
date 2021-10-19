# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.models.uniter import UniterImageEmbeddings
from omegaconf import OmegaConf


class TestUniterImageEmbeddings(unittest.TestCase):
    def test_forward_has_correct_output_dim(self):
        bs = 32
        num_feat = 100
        config = OmegaConf.create({"img_dim": 1024, "hidden_size": 256, "pos_dim": 7})
        embedding = UniterImageEmbeddings(config)
        img_feat = torch.rand((bs, num_feat, config["img_dim"]))
        img_pos_feat = torch.rand((bs, num_feat, config["pos_dim"]))
        type_embeddings = torch.ones((bs, num_feat, 1), dtype=torch.long)

        output = embedding(img_feat, img_pos_feat, type_embeddings, img_masks=None)
        self.assertEquals(list(output.shape), [32, 100, 256])
