# Copyright (c) Facebook, Inc. and its affiliates.

import io
import unittest

import torch
from mmf.common.registry import registry
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports

import tests.test_utils as test_utils


class TestVisualBertTorchscript(unittest.TestCase):
    def setUp(self):
        setup_imports()
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
    def test_load_save_pretrain_model(self):
        self.pretrain_model.model.eval()
        script_model = torch.jit.script(self.pretrain_model.model)
        buffer = io.BytesIO()
        torch.jit.save(script_model, buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)
        self.assertTrue(test_utils.assertModulesEqual(script_model, loaded_model))

    @test_utils.skip_if_no_network
    def test_pretrained_model(self):
        self.pretrain_model.model.eval()

        input_ids = torch.randint(low=0, high=30255, size=(1, 128)).long()
        input_mask = torch.ones((1, 128)).long()
        attention_mask = torch.ones((1, 228)).long()
        token_type_ids = torch.zeros(1, 128).long()
        visual_embeddings = torch.rand((1, 100, 2048)).float()
        visual_embeddings_type = torch.zeros(1, 100).long()
        masked_lm_labels = torch.zeros((1, 228), dtype=torch.long).fill_(-1)

        self.pretrain_model.eval()

        with torch.no_grad():
            model_output = self.pretrain_model.model(
                input_ids=input_ids,
                input_mask=input_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_embeddings=visual_embeddings,
                visual_embeddings_type=visual_embeddings_type,
                masked_lm_labels=masked_lm_labels,
            )
        script_model = torch.jit.script(self.pretrain_model.model)
        with torch.no_grad():
            script_output = script_model(
                input_ids=input_ids,
                input_mask=input_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_embeddings=visual_embeddings,
                visual_embeddings_type=visual_embeddings_type,
                masked_lm_labels=masked_lm_labels,
            )
        self.assertEqual(
            model_output["masked_lm_loss"], script_output["masked_lm_loss"]
        )

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
        attention_mask = torch.ones((1, 228)).long()
        token_type_ids = torch.zeros(1, 128).long()
        visual_embeddings = torch.rand((1, 100, 2048)).float()
        visual_embeddings_type = torch.zeros(1, 100).long()

        with torch.no_grad():
            model_output = self.finetune_model.model(
                input_ids=input_ids,
                input_mask=input_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_embeddings=visual_embeddings,
                visual_embeddings_type=visual_embeddings_type,
            )

        script_model = torch.jit.script(self.finetune_model.model)
        with torch.no_grad():
            script_output = script_model(
                input_ids=input_ids,
                input_mask=input_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_embeddings=visual_embeddings,
                visual_embeddings_type=visual_embeddings_type,
            )

        self.assertTrue(torch.equal(model_output["scores"], script_output["scores"]))
