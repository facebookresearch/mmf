# Copyright (c) Facebook, Inc. and its affiliates.

import io
import unittest

import torch
from mmf.common.registry import registry
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports

import tests.test_utils as test_utils


class TestMMBTTorchscript(unittest.TestCase):
    def setUp(self):
        setup_imports()
        model_name = "mmbt"
        args = test_utils.dummy_args(model=model_name)
        configuration = Configuration(args)
        config = configuration.get_config()
        model_class = registry.get_model_class(model_name)
        config.model_config[model_name]["training_head_type"] = "classification"
        config.model_config[model_name]["num_labels"] = 2
        self.finetune_model = model_class(config.model_config[model_name])
        self.finetune_model.build()

    @test_utils.skip_if_no_network
    def test_load_save_finetune_model(self):
        self.finetune_model.model.eval()
        script_model = torch.jit.script(self.finetune_model.model)
        buffer = io.BytesIO()
        torch.jit.save(script_model, buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)
        self.assertTrue(test_utils.assertModulesEqual(script_model, loaded_model))

    @test_utils.skip_if_no_network
    def test_finetune_model(self):
        self.finetune_model.model.eval()
        input_ids = torch.randint(low=0, high=30255, size=(1, 128)).long()
        input_mask = torch.ones((1, 128)).long()
        segment_ids = torch.zeros(1, 128).long()
        visual_embeddings = torch.rand((1, 3, 300, 300)).float()

        with torch.no_grad():
            model_output = self.finetune_model.model(
                input_modal=visual_embeddings,
                input_ids=input_ids,
                segment_ids=segment_ids,
                input_mask=input_mask,
            )

        script_model = torch.jit.script(self.finetune_model.model)
        with torch.no_grad():
            script_output = script_model(
                input_modal=visual_embeddings,
                input_ids=input_ids,
                segment_ids=segment_ids,
                input_mask=input_mask,
            )

        self.assertTrue(torch.equal(model_output["scores"], script_output["scores"]))
