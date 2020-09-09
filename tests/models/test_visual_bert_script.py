# Copyright (c) Facebook, Inc. and its affiliates.

import io
import itertools
import unittest

import torch
from mmf.common.registry import registry
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports

from tests.test_utils import dummy_args


class TestVisualBertTorchscript(unittest.TestCase):
    def setUp(self):
        setup_imports()
        model_name = "visual_bert"
        args = dummy_args(model=model_name)
        configuration = Configuration(args)
        config = configuration.get_config()
        model_class = registry.get_model_class(model_name)
        self.pretrain_model = model_class(config.model_config[model_name])
        self.pretrain_model.build()

        config.model_config[model_name]["training_head_type"] = "classification"
        config.model_config[model_name]["num_labels"] = 2
        self.finetune_model = model_class(config.model_config[model_name])
        self.finetune_model.build()

    def assertModulesEqual(self, mod1, mod2, message=None):
        for p1, p2 in itertools.zip_longest(mod1.parameters(), mod2.parameters()):
            self.assertTrue(p1.equal(p2), message)

    def test_load_save_pretrain_model(self):
        self.pretrain_model.model.eval()
        script_model = torch.jit.script(self.pretrain_model.model)
        buffer = io.BytesIO()
        torch.jit.save(script_model, buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)
        self.assertModulesEqual(script_model, loaded_model)

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

    def test_load_save_finetune_model(self):
        self.finetune_model.model.eval()
        script_model = torch.jit.script(self.finetune_model.model)
        buffer = io.BytesIO()
        torch.jit.save(script_model, buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)
        self.assertModulesEqual(script_model, loaded_model)

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
