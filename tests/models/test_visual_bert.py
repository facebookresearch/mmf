# Copyright (c) Facebook, Inc. and its affiliates.

import io
import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.registry import registry
from mmf.common.sample import SampleList
from mmf.modules.hf_layers import replace_with_jit
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports


class TestVisualBertTorchscript(unittest.TestCase):
    def setUp(self):
        setup_imports()
        replace_with_jit()
        model_name = "visual_bert"
        args = test_utils.dummy_args(model=model_name)
        configuration = Configuration(args)
        config = configuration.get_config()
        model_class = registry.get_model_class(model_name)
        self.pretrain_model = model_class(config.model_config[model_name])
        self.pretrain_model.build()

        config.model_config[model_name]["training_head_type"] = "classification"
        config.model_config[model_name]["num_labels"] = 2
        self.finetune_model = model_class(config.model_config[model_name])
        self.finetune_model.build()

    @test_utils.skip_if_no_network
    def test_pretrained_model_jit_assertion(self):
        self.pretrain_model.eval()
        self.assertRaises(RuntimeError, torch.jit.script(self.pretrain_model))

    @test_utils.skip_if_no_network
    def test_pretrained_model(self):
        self.pretrain_model.eval()

        sample_list = SampleList()

        sample_list.add_field(
            "input_ids", torch.randint(low=0, high=30255, size=(1, 128)).long()
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

    @test_utils.skip_if_no_network
    def test_load_save_finetune_model(self):
        self.finetune_model.eval()
        script_model = torch.jit.script(self.finetune_model)
        buffer = io.BytesIO()
        torch.jit.save(script_model, buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)
        self.assertTrue(test_utils.assertModulesEqual(script_model, loaded_model))

    @test_utils.skip_if_no_network
    def test_finetune_model(self):
        self.finetune_model.eval()
        sample_list = SampleList()

        sample_list.add_field(
            "input_ids", torch.randint(low=0, high=30255, size=(1, 128)).long()
        )
        sample_list.add_field("input_mask", torch.ones((1, 128)).long())
        sample_list.add_field("segment_ids", torch.zeros(1, 128).long())
        sample_list.add_field("image_feature_0", torch.rand((1, 100, 2048)).float())

        with torch.no_grad():
            model_output = self.finetune_model(sample_list)

        script_model = torch.jit.script(self.finetune_model)
        with torch.no_grad():
            script_output = script_model(sample_list)

        self.assertTrue(torch.equal(model_output["scores"], script_output["scores"]))
