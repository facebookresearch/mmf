# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.models.vinvl import (
    BertImgModel,
    VinVLForClassification,
    VinVLForPretraining,
)
from mmf.utils.general import get_current_device
from transformers.modeling_bert import BertConfig


class TestVinVLBertImageModel(unittest.TestCase):
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
        model = BertImgModel(bert_config)

        model.eval()
        model = model.to(get_current_device())

        bs = 8
        num_feats = 70
        max_sentence_len = 25
        input_ids = torch.ones((bs, max_sentence_len), dtype=torch.long)
        img_feat = torch.rand((bs, num_feats, img_feature_dim))

        with torch.no_grad():
            model_output = model(input_ids, img_feat).final_layer
        self.assertEqual(model_output.shape, torch.Size([8, 95, 768]))


class TestVinVLForClassification(unittest.TestCase):
    def test_forward(self):
        model = VinVLForClassification().to(get_current_device())
        model.eval()

        bs = 8
        num_feats = 70
        max_sentence_len = 25
        img_feature_dim = 2054
        input_ids = torch.ones((bs, max_sentence_len), dtype=torch.long)
        img_feats = torch.rand((bs, num_feats, img_feature_dim))
        attention_mask = torch.ones(
            (bs, max_sentence_len + num_feats), dtype=torch.long
        )
        token_type_ids = torch.zeros_like(input_ids)
        labels = torch.ones((bs, 1)).long()

        with torch.no_grad():
            model_output = model(
                input_ids=input_ids,
                img_feats=img_feats,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
        self.assertTrue("losses" in model_output)
        self.assertTrue("scores" in model_output)
        self.assertTrue("ce" in model_output["losses"])


class TestVinVLForPretraining(unittest.TestCase):
    def test_forward(self):
        model = VinVLForPretraining().to(get_current_device())
        model.eval()

        bs = 8
        num_feats = 70
        max_sentence_len = 25
        img_feature_dim = 2054
        input_ids = torch.ones((bs, max_sentence_len), dtype=torch.long)
        img_feats = torch.rand((bs, num_feats, img_feature_dim))
        attention_mask = torch.ones(
            (bs, max_sentence_len + num_feats), dtype=torch.long
        )
        token_type_ids = torch.zeros_like(input_ids)
        input_ids_masked = input_ids
        lm_label_ids = -torch.ones_like(input_ids).long()
        contrastive_labels = torch.zeros((bs, 1)).long()

        with torch.no_grad():
            model_output = model(
                img_feats=img_feats,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                input_ids_masked=input_ids_masked,
                lm_label_ids=lm_label_ids,
                contrastive_labels=contrastive_labels,
                input_ids_corrupt=input_ids,
                token_type_ids_corrupt=token_type_ids,
                attention_mask_corrupt=attention_mask,
            )
        self.assertTrue("losses" in model_output)
        self.assertTrue("masked_lm_loss" in model_output["losses"])
        self.assertTrue("vinvl_three_way_contrastive" in model_output["losses"])
