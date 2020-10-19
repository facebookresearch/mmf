# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.sample import SampleList
from mmf.modules.hf_layers import replace_with_jit
from mmf.utils.build import build_model
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports


BERT_VOCAB_SIZE = 30255


class TestVisualBertTorchscript(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        replace_with_jit()
        model_name = "visual_bert"
        args = test_utils.dummy_args(model=model_name)
        configuration = Configuration(args)
        config = configuration.get_config()
        model_config = config.model_config[model_name]
        model_config["training_head_type"] = "classification"
        model_config["num_labels"] = 2
        model_config.model = model_name
        self.finetune_model = build_model(model_config)

    def test_load_save_finetune_model(self):
        self.assertTrue(test_utils.verify_torchscript_models(self.finetune_model))

    def test_finetune_model(self):
        model = self.finetune_model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(
                model, vocab_size=BERT_VOCAB_SIZE
            )
        )


class TestVisualBertPretraining(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        replace_with_jit()
        model_name = "visual_bert"
        args = test_utils.dummy_args(model=model_name)
        configuration = Configuration(args)
        config = configuration.get_config()
        model_config = config.model_config[model_name]
        model_config.model = model_name
        self.pretrain_model = build_model(model_config)

    def test_pretrained_model(self):
        sample_list = SampleList()

        sample_list.add_field(
            "input_ids",
            torch.randint(low=0, high=BERT_VOCAB_SIZE, size=(1, 128)).long(),
        )
        sample_list.add_field("input_mask", torch.ones((1, 128)).long())
        sample_list.add_field("segment_ids", torch.zeros(1, 128).long())
        sample_list.add_field("image_feature_0", torch.rand((1, 100, 2048)).float())
        sample_list.add_field(
            "lm_label_ids", torch.zeros((1, 128), dtype=torch.long).fill_(-1)
        )

        self.pretrain_model.eval()

        sample_list.dataset_name = "random"
        sample_list.dataset_type = "test"
        with torch.no_grad():
            model_output = self.pretrain_model(sample_list)

        self.assertTrue("losses" in model_output)
        self.assertTrue("random/test/masked_lm_loss" in model_output["losses"])
        self.assertTrue(model_output["losses"]["random/test/masked_lm_loss"] == 0)
