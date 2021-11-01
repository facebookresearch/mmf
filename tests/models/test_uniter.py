# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
from mmf.models.uniter import UNITERImageEmbeddings, UNITERModelBase
from mmf.utils.general import get_current_device
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


class TestUNITERModelBase(unittest.TestCase):
    def test_pretrained_model(self):
        img_dim = 1024
        model = UNITERModelBase(img_dim=img_dim)

        model.eval()
        model = model.to(get_current_device())

        bs = 8
        num_feats = 100
        max_sentence_len = 25
        pos_dim = 7
        input_ids = torch.ones((bs, max_sentence_len), dtype=torch.long)
        img_feat = torch.rand((bs, num_feats, img_dim))
        img_pos_feat = torch.rand((bs, num_feats, pos_dim))
        position_ids = torch.arange(
            0, input_ids.size(1), dtype=torch.long, device=img_feat.device
        ).unsqueeze(0)
        attention_mask = torch.ones((bs, max_sentence_len + num_feats))

        with torch.no_grad():
            model_output = model(
                input_ids, position_ids, img_feat, img_pos_feat, attention_mask
            )

        self.assertEqual(model_output.shape, torch.Size([8, 125, 768]))
