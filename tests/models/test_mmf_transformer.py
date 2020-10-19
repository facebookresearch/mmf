# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import tests.test_utils as test_utils
from mmf.utils.build import build_model
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports


BERT_VOCAB_SIZE = 30255
ROBERTA_VOCAB_SIZE = 50265
XLM_ROBERTA_VOCAB_SIZE = 250002


class TestMMFTransformerTorchscript(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        self.model_name = "mmf_transformer"
        args = test_utils.dummy_args(model=self.model_name)
        configuration = Configuration(args)
        self.config = configuration.get_config()
        self.config.model_config[self.model_name].model = self.model_name
        self.finetune_model = build_model(self.config.model_config[self.model_name])

    def test_load_save_finetune_model(self):
        self.assertTrue(test_utils.verify_torchscript_models(self.finetune_model))

    def test_finetune_bert_base(self):
        model = self.finetune_model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(
                model, vocab_size=BERT_VOCAB_SIZE
            )
        )

    def test_finetune_roberta_base(self):
        self.config.model_config[self.model_name]["transformer_base"] = "roberta-base"
        model = build_model(self.config.model_config[self.model_name])
        model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(
                model, vocab_size=ROBERTA_VOCAB_SIZE
            )
        )

    @test_utils.skip_if_no_network
    def test_finetune_xlmr_base(self):
        self.config.model_config[self.model_name][
            "transformer_base"
        ] = "xlm-roberta-base"
        model = build_model(self.config.model_config[self.model_name])
        model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(
                model, vocab_size=XLM_ROBERTA_VOCAB_SIZE
            )
        )
