# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.models.uniter import UNITERImageEmbeddings
from omegaconf import OmegaConf


class TestUNITERImageEmbeddings(unittest.TestCase):
    def setUp(self):
        bs = 32
        num_feat = 100
        self.config = OmegaConf.create(
            {"img_dim": 1024, "hidden_size": 256, "pos_dim": 7}
        )
        self.img_feat = torch.rand((bs, num_feat, self.config["img_dim"]))
        self.img_pos_feat = torch.rand((bs, num_feat, self.config["pos_dim"]))
        self.type_embeddings = torch.ones((bs, num_feat, 1), dtype=torch.long)

    def test_forward(self):
        embedding = UNITERImageEmbeddings(**self.config)
        output = embedding(
            self.img_feat, self.img_pos_feat, self.type_embeddings, img_masks=None
        )
        self.assertEquals(list(output.shape), [32, 100, 256])
