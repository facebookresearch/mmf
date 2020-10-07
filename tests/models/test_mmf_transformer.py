# Copyright (c) Facebook, Inc. and its affiliates.

import io
import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports


class TestMMFTransformerTorchscript(unittest.TestCase):
    def setUp(self):
        setup_imports()
        self.model_name = "mmf_transformer"
        args = test_utils.dummy_args(model=self.model_name)
        configuration = Configuration(args)
        self.config = configuration.get_config()
        self.model_class = registry.get_model_class(self.model_name)
        self.finetune_model = self.model_class(
            self.config.model_config[self.model_name]
        )
        self.finetune_model.build()

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
    def test_finetune_bert_base(self):
        self.finetune_model.eval()
        test_sample = Sample()
        test_sample.input_ids = torch.randint(low=0, high=30255, size=(128,)).long()
        test_sample.input_mask = torch.ones(128).long()
        test_sample.segment_ids = torch.zeros(128).long()
        test_sample.image = torch.rand((3, 300, 300)).float()
        test_sample_list = SampleList([test_sample])

        with torch.no_grad():
            model_output = self.finetune_model(test_sample_list)

        script_model = torch.jit.script(self.finetune_model)
        with torch.no_grad():
            script_output = script_model(test_sample_list)

        self.assertTrue(torch.equal(model_output["scores"], script_output["scores"]))

    @test_utils.skip_if_no_network
    def test_finetune_roberta_base(self):
        self.config.model_config[self.model_name]["transformer_base"] = "roberta-base"
        model = self.model_class(self.config.model_config[self.model_name])
        model.build()
        model.eval()
        test_sample = Sample()
        test_sample.input_ids = torch.randint(low=0, high=50265, size=(128,)).long()
        test_sample.input_mask = torch.ones(128).long()
        test_sample.segment_ids = torch.zeros(128).long()
        test_sample.image = torch.rand((3, 300, 300)).float()
        test_sample_list = SampleList([test_sample])

        with torch.no_grad():
            model_output = model(test_sample_list)

        script_model = torch.jit.script(model)
        with torch.no_grad():
            script_output = script_model(test_sample_list)

        self.assertTrue(torch.equal(model_output["scores"], script_output["scores"]))

    @test_utils.skip_if_no_network
    def test_finetune_xlmr_base(self):
        self.config.model_config[self.model_name][
            "transformer_base"
        ] = "xlm-roberta-base"
        model = self.model_class(self.config.model_config[self.model_name])
        model.build()
        model.eval()
        test_sample = Sample()
        test_sample.input_ids = torch.randint(low=0, high=250002, size=(128,)).long()
        test_sample.input_mask = torch.ones(128).long()
        test_sample.segment_ids = torch.zeros(128).long()
        test_sample.image = torch.rand((3, 300, 300)).float()
        test_sample_list = SampleList([test_sample])

        with torch.no_grad():
            model_output = model(test_sample_list)

        script_model = torch.jit.script(model)
        with torch.no_grad():
            script_output = script_model(test_sample_list)

        self.assertTrue(torch.equal(model_output["scores"], script_output["scores"]))
