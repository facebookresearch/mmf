# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.sample import SampleList
from mmf.models.vinvl import VinVLBase, VinVLForClassification, VinVLForPretraining
from mmf.utils.build import build_model
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports, teardown_imports
from mmf.utils.general import get_current_device
from omegaconf import OmegaConf

try:
    from transformers3.modeling_bert import BertConfig
except ImportError:
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
        self.assertTrue("three_way_contrastive_loss" in model_output["losses"])


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
        self.classification_config = OmegaConf.create(
            {**model_config, **classification_config_dict}
        )

        pretraining_config_dict = {
            "do_pretraining": True,
            "heads": {"mlm": {"hidden_size": 768}},
        }
        self.pretraining_config = OmegaConf.create(
            {**model_config, **pretraining_config_dict}
        )

        self.sample_list = self._get_sample_list()

    def tearDown(self):
        teardown_imports()

    def _get_sample_list(self):
        bs = 8
        num_feats = 70

        class MockObj:
            pass

        mock_input = MockObj()
        mock_vinvl_input_tensors(mock_input, bs=bs, num_feats=num_feats)

        input_mask = torch.ones_like(mock_input.input_ids)
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

        sample_list = SampleList()
        sample_list.add_field("input_ids", mock_input.input_ids)
        sample_list.add_field("input_ids_corrupt", mock_input.input_ids)
        sample_list.add_field("input_ids_masked", mock_input.input_ids)
        sample_list.add_field("image_feature_0", mock_input.img_feats)
        sample_list.add_field("image_info_0", image_info)
        sample_list.add_field("input_mask", input_mask)
        sample_list.add_field("input_mask_corrupt", input_mask)
        sample_list.add_field("segment_ids", mock_input.token_type_ids)
        sample_list.add_field("segment_ids_corrupt", mock_input.token_type_ids)
        sample_list.add_field("labels", mock_input.labels)
        sample_list.add_field("contrastive_labels", mock_input.contrastive_labels)
        sample_list.add_field("lm_label_ids", mock_input.lm_label_ids)
        sample_list = sample_list.to(get_current_device())
        sample_list.dataset_name = "test"
        sample_list.dataset_type = "test"
        return sample_list

    def test_vinvl_for_classification(self):
        model_for_classification = build_model(self.classification_config)
        model_for_classification.eval()
        model_for_classification = model_for_classification.to(get_current_device())
        with torch.no_grad():
            model_output = model_for_classification(self.sample_list)

        self.assertTrue("losses" in model_output)
        self.assertTrue("ce" in model_output["losses"])

    def test_vinvl_for_pretraining(self):
        model_for_pretraining = build_model(self.pretraining_config)
        model_for_pretraining.eval()
        model_for_pretraining = model_for_pretraining.to(get_current_device())

        with torch.no_grad():
            model_output = model_for_pretraining(self.sample_list)

        self.assertTrue("losses" in model_output)
        self.assertTrue("masked_lm_loss" in model_output["losses"])
        self.assertTrue("three_way_contrastive_loss" in model_output["losses"])
