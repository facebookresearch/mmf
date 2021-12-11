# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.models.vinvl import (
    VinVLBase,
)
from mmf.utils.general import get_current_device
from transformers.modeling_bert import BertConfig


class TestVinVLBase(unittest.TestCase):
    def test_forward(self):
        img_feature_dim = 2054
        bert_model_name = "bert-base-uncased"
        use_img_layernorm = True
        img_layer_norm_eps = 1e-12
        bert_config = BertConfig.from_pretrained(bert_model_name)
        # augment hf BertConfig for vinvl BertImgModel config
        bert_config.img_feature_dim = img_feature_dim
        bert_config.use_img_layernorm = use_img_layernorm
        bert_config.img_layer_norm_eps = img_layer_norm_eps
        model = VinVLBase(bert_config)

        model.eval()
        model = model.to(get_current_device())

        bs = 8
        num_feats = 70
        max_sentence_len = 25
        input_ids = torch.ones((bs, max_sentence_len), dtype=torch.long)
        img_feat = torch.rand((bs, num_feats, img_feature_dim))

        with torch.no_grad():
            model_output = model(input_ids, img_feat).last_hidden_state
        self.assertEqual(model_output.shape, torch.Size([8, 95, 768]))
