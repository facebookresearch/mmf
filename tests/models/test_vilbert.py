# Copyright (c) Facebook, Inc. and its affiliates.

import gc
import unittest

import tests.test_utils as test_utils
import torch
from mmf.modules.hf_layers import undo_replace_with_jit
from mmf.utils.build import build_model
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports, teardown_imports


BERT_VOCAB_SIZE = 30255


class TestViLBertTorchscript(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        model_name = "vilbert"
        args = test_utils.dummy_args(model=model_name)
        configuration = Configuration(args)
        config = configuration.get_config()
        self.vision_feature_size = 1024
        self.vision_target_size = 1279
        model_config = config.model_config[model_name]
        model_config["training_head_type"] = "pretraining"
        model_config["visual_embedding_dim"] = self.vision_feature_size
        model_config["v_feature_size"] = self.vision_feature_size
        model_config["v_target_size"] = self.vision_target_size
        model_config["dynamic_attention"] = False
        model_config.model = model_name

        model_config["training_head_type"] = "classification"
        model_config["num_labels"] = 2
        self.model_config = model_config

    def tearDown(self):
        teardown_imports()
        undo_replace_with_jit()
        del self.model_config
        gc.collect()

    def test_load_save_pretrain_model(self):
        self.model_config["training_head_type"] = "pretraining"
        pretrain_model = build_model(self.model_config)
        self.assertTrue(test_utils.verify_torchscript_models(pretrain_model.model))

    def test_load_save_finetune_model(self):
        self.model_config["training_head_type"] = "classification"
        finetune_model = build_model(self.model_config)
        self.assertTrue(test_utils.verify_torchscript_models(finetune_model.model))

    def test_pretrained_model(self):
        self.model_config["training_head_type"] = "pretraining"
        pretrain_model = build_model(self.model_config)
        pretrain_model.model.eval()
        num_bbox_per_image = 10
        input_ids = torch.randint(low=0, high=BERT_VOCAB_SIZE, size=(1, 128)).long()
        attention_mask = torch.ones((1, 128)).long()
        token_type_ids = torch.zeros(1, 128).long()
        visual_embeddings = torch.rand(
            (1, num_bbox_per_image, self.vision_feature_size)
        ).float()
        image_attention_mask = torch.zeros((1, num_bbox_per_image)).long()
        visual_locations = torch.rand((1, num_bbox_per_image, 5)).float()
        masked_lm_labels = torch.zeros((1, 128), dtype=torch.long).fill_(-1)
        image_target = torch.zeros(1, num_bbox_per_image, self.vision_target_size)
        image_label = torch.ones(1, num_bbox_per_image).fill_(-1)
        pretrain_model.eval()

        with torch.no_grad():
            model_output = pretrain_model.model(
                input_ids=input_ids,
                image_feature=visual_embeddings,
                image_location=visual_locations,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                image_attention_mask=image_attention_mask,
                masked_lm_labels=masked_lm_labels,
                image_label=image_label,
                image_target=image_target,
            )
        script_model = torch.jit.script(pretrain_model.model)
        with torch.no_grad():
            script_output = script_model(
                input_ids=input_ids,
                image_feature=visual_embeddings,
                image_location=visual_locations,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                image_attention_mask=image_attention_mask,
                masked_lm_labels=masked_lm_labels,
                image_label=image_label,
                image_target=image_target,
            )
        self.assertEqual(
            torch.isnan(model_output["masked_lm_loss"]),
            torch.isnan(script_output["masked_lm_loss"]),
        )

    def test_finetune_model(self):
        self.model_config["training_head_type"] = "classification"
        finetune_model = build_model(self.model_config)
        finetune_model.model.eval()
        num_bbox_per_image = 10
        input_ids = torch.randint(low=0, high=BERT_VOCAB_SIZE, size=(1, 128)).long()
        attention_mask = torch.ones((1, 128)).long()
        token_type_ids = torch.zeros(1, 128).long()
        visual_embeddings = torch.rand(
            (1, num_bbox_per_image, self.vision_feature_size)
        ).float()
        image_attention_mask = torch.zeros((1, num_bbox_per_image)).long()
        visual_locations = torch.rand((1, num_bbox_per_image, 5)).float()
        finetune_model.eval()

        with torch.no_grad():
            model_output = finetune_model.model(
                input_ids=input_ids,
                image_feature=visual_embeddings,
                image_location=visual_locations,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                image_attention_mask=image_attention_mask,
            )
        script_model = torch.jit.script(finetune_model.model)
        with torch.no_grad():
            script_output = script_model(
                input_ids=input_ids,
                image_feature=visual_embeddings,
                image_location=visual_locations,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                image_attention_mask=image_attention_mask,
            )
        self.assertTrue(torch.equal(model_output["scores"], script_output["scores"]))
