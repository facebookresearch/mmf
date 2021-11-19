# Copyright (c) Facebook, Inc. and its affiliates.

import gc
import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.sample import SampleList
from mmf.models.vinvl import (
    BertImgModel,
    VinVLForClassification,
    VinVLForPretraining,
)
from mmf.utils.build import build_model
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports, teardown_imports
from mmf.utils.general import get_current_device
from omegaconf import OmegaConf
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


class TestVinVLModel(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        model_name = "vinvl"
        args = test_utils.dummy_args(model=model_name, dataset="test")
        configuration = Configuration(args)
        config = configuration.get_config()
        model_config = config.model_config[model_name]
        model_config.model = model_name
        model_config.do_pretraining = False
        classification_config_dict = {
            "do_pretraining": False,
            "heads": {"mlp": {"num_labels": 3129}},
            "ce_loss": {"ignore_index": -1},
        }
        classification_config = OmegaConf.create(
            {**model_config, **classification_config_dict}
        )

        pretraining_config_dict = {
            "do_pretraining": True,
            "heads": {"mlm": {"hidden_size": 768}},
        }
        pretraining_config = OmegaConf.create(
            {**model_config, **pretraining_config_dict}
        )

        self.model_for_classification = build_model(classification_config)
        self.model_for_pretraining = build_model(pretraining_config)

    def tearDown(self):
        teardown_imports()
        del self.model_for_classification
        del self.model_for_pretraining
        gc.collect()

    def _get_sample_list(self):
        bs = 8
        num_feats = 70
        max_sentence_len = 25
        img_feature_dim = 2054
        input_ids = torch.ones((bs, max_sentence_len), dtype=torch.long)
        img_feats = torch.rand((bs, num_feats, img_feature_dim))

        max_features = torch.ones((bs, num_feats)) * num_feats
        bbox = torch.randint(50, 200, (bs, num_feats, 4)).float()
        image_height = torch.randint(100, 300, (bs,))
        image_width = torch.randint(100, 300, (bs,))
        image_info = {
            "max_features": max_features,
            "bbox": bbox,
            "image_height": image_height,
            "image_width": image_width,
        }

        token_type_ids = torch.zeros_like(input_ids)
        input_ids_masked = input_ids
        lm_label_ids = -torch.ones_like(input_ids).long()
        contrastive_labels = torch.zeros((bs, 1)).long()
        input_mask = torch.ones((bs, max_sentence_len), dtype=torch.long)

        labels = torch.zeros((bs, 1)).long()

        sample_list = SampleList()
        sample_list.add_field("input_ids", input_ids)
        sample_list.add_field("input_ids_corrupt", input_ids)
        sample_list.add_field("input_ids_masked", input_ids_masked)
        sample_list.add_field("image_feature_0", img_feats)
        sample_list.add_field("image_info_0", image_info)
        sample_list.add_field("input_mask", input_mask)
        sample_list.add_field("input_mask_corrupt", input_mask)
        sample_list.add_field("segment_ids", token_type_ids)
        sample_list.add_field("segment_ids_corrupt", token_type_ids)
        sample_list.add_field("labels", labels)
        sample_list.add_field("contrastive_labels", contrastive_labels)
        sample_list.add_field("lm_label_ids", lm_label_ids)
        sample_list = sample_list.to(get_current_device())
        return sample_list

    def test_vinvl_for_classification(self):
        self.model_for_classification.eval()
        self.model_for_classification = self.model_for_classification.to(
            get_current_device()
        )
        sample_list = self._get_sample_list()

        sample_list.dataset_name = "test"
        sample_list.dataset_type = "test"
        with torch.no_grad():
            model_output = self.model_for_classification(sample_list)

        self.assertTrue("losses" in model_output)
        self.assertTrue("ce" in model_output["losses"])

    def test_vinvl_for_pretraining(self):
        self.model_for_pretraining.eval()
        self.model_for_pretraining = self.model_for_pretraining.to(get_current_device())
        sample_list = self._get_sample_list()

        sample_list.dataset_name = "test"
        sample_list.dataset_type = "test"
        with torch.no_grad():
            model_output = self.model_for_pretraining(sample_list)

        self.assertTrue("losses" in model_output)
        self.assertTrue("masked_lm_loss" in model_output["losses"])
        self.assertTrue("vinvl_three_way_contrastive" in model_output["losses"])
