# Copyright (c) Facebook, Inc. and its affiliates.
import gc
import unittest

import torch
from mmf.models.uniter import UniterImageEmbeddings, UniterModelBase
from mmf.utils.general import get_current_device
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


class TestUniterModelBase(unittest.TestCase):
    def tearDown(self):
        del self.model
        gc.collect()

    def test_pretrained_model(self):
        img_dim = 1024
        config = OmegaConf.create({"image_embeddings": {"img_dim": img_dim}})
        self.model = UniterModelBase(config)

        self.model.eval()
        self.model = self.model.to(get_current_device())

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
            model_output = self.model(
                input_ids, position_ids, img_feat, img_pos_feat, attention_mask
            )

        self.assertEqual(model_output.shape, torch.Size([8, 125, 768]))
