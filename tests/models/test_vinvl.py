# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.models.vinvl import (
    VinVLBase,
    VinVLForClassification,
    VinVLForPretraining,
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


def mock_vinvl_input_tensors(
    cls, bs=8, num_feats=70, max_sentence_len=25, img_feature_dim=2054
):
    cls.input_ids = torch.ones((bs, max_sentence_len), dtype=torch.long)
    cls.img_feats = torch.rand((bs, num_feats, img_feature_dim))
    cls.attention_mask = torch.ones(
        (bs, max_sentence_len + num_feats), dtype=torch.long
    )
    cls.token_type_ids = torch.zeros_like(cls.input_ids)
    cls.labels = torch.ones((bs, 1)).long()

    cls.lm_label_ids = -torch.ones_like(cls.input_ids).long()
    cls.contrastive_labels = torch.zeros((bs, 1)).long()


class TestVinVLForClassificationAndPretraining(unittest.TestCase):
    def setUp(self):
        mock_vinvl_input_tensors(self)

    def test_classification_forward(self):
        model = VinVLForClassification().to(get_current_device())
        model.eval()

        with torch.no_grad():
            model_output = model(
                input_ids=self.input_ids,
                img_feats=self.img_feats,
                attention_mask=self.attention_mask,
                token_type_ids=self.token_type_ids,
                labels=self.labels,
            )
        self.assertTrue("losses" in model_output)
        self.assertTrue("scores" in model_output)
        self.assertTrue("ce" in model_output["losses"])

    def test_pretraining_forward(self):
        model = VinVLForPretraining().to(get_current_device())
        model.eval()

        with torch.no_grad():
            model_output = model(
                img_feats=self.img_feats,
                attention_mask=self.attention_mask,
                token_type_ids=self.token_type_ids,
                input_ids_masked=self.input_ids,
                lm_label_ids=self.lm_label_ids,
                contrastive_labels=self.contrastive_labels,
                input_ids_corrupt=self.input_ids,
                token_type_ids_corrupt=self.token_type_ids,
                attention_mask_corrupt=self.attention_mask,
            )
        self.assertTrue("losses" in model_output)
        self.assertTrue("masked_lm_loss" in model_output["losses"])
        self.assertTrue("vinvl_three_way_contrastive_loss" in model_output["losses"])
