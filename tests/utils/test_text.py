# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest

import mmf.utils.text as text_utils
import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports
from mmf.utils.general import get_mmf_root

from tests.test_utils import dummy_args
from tests.utils.test_model import TestDecoderModel


class TestUtilsText(unittest.TestCase):
    TOKENS = ["this", "will", "be", "a", "test", "of", "tokens"]
    TOKENIZE_EXAMPLE = "This will be a test of tokens?"
    VOCAB_EXAMPLE_SENTENCES = [
        "Are there more big green things than large purple shiny cubes?"
        "How many other things are there of the same shape as the tiny "
        + "cyan matte object?",
        "Is the color of the large sphere the same as the large matte cube?"
        "What material is the big object that is right of the brown cylinder and "
        "left of the large brown sphere?",
        "How big is the brown shiny sphere? ;",
    ]

    def setUp(self):
        setup_imports()
        torch.manual_seed(1234)
        config_path = os.path.join(
            get_mmf_root(),
            "..",
            "projects",
            "butd",
            "configs",
            "coco",
            "nucleus_sampling.yaml",
        )
        config_path = os.path.abspath(config_path)
        args = dummy_args(model="butd", dataset="coco")
        args.opts.append(f"config={config_path}")
        configuration = Configuration(args)
        configuration.config.datasets = "coco"
        configuration.config.model_config.butd.inference.params.sum_threshold = 0.5
        configuration.freeze()
        self.config = configuration.config
        registry.register("config", self.config)

    def test_tokenize(self):
        tokens = text_utils.tokenize(self.TOKENIZE_EXAMPLE)

        self.assertEqual(list(tokens), self.TOKENS)

    def test_generate_ngrams(self):
        ngrams = text_utils.generate_ngrams(self.TOKENS, 2)

        self.assertEqual(
            list(ngrams),
            ["this will", "will be", "be a", "a test", "test of", "of tokens"],
        )

        ngrams = text_utils.generate_ngrams(self.TOKENS, 3)

        self.assertEqual(
            list(ngrams),
            ["this will be", "will be a", "be a test", "a test of", "test of tokens"],
        )

    def test_generate_ngrams_range(self):
        # Test generation of 1grams to 3gram
        ngrams = text_utils.generate_ngrams_range(self.TOKENS, (1, 4))

        expected_ngrams = self.TOKENS + [
            "this will",
            "will be",
            "be a",
            "a test",
            "test of",
            "of tokens",
            "this will be",
            "will be a",
            "be a test",
            "a test of",
            "test of tokens",
        ]

        self.assertEqual(list(ngrams), expected_ngrams)

    def test_vocab_from_text(self):
        vocab = text_utils.VocabFromText(self.VOCAB_EXAMPLE_SENTENCES)

        self.assertEqual(vocab.get_size(), 41)
        self.assertEqual(len(vocab), 41)
        self.assertEqual(vocab.get_unk_index(), 1)

        self.assertEqual(vocab.itos[0], vocab.DEFAULT_TOKENS[0])
        self.assertEqual(vocab.itos[34], "that")
        self.assertEqual(vocab.itos[31], "cube")
        self.assertEqual(vocab.itos[25], "cyan")
        self.assertEqual(vocab.itos[20], "the")
        self.assertEqual(vocab.itos[10], "than")

        self.assertEqual(vocab.stoi["sphere"], 30)
        self.assertEqual(vocab.stoi["shape"], 22)

        vocab = text_utils.VocabFromText(self.VOCAB_EXAMPLE_SENTENCES, min_count=10)
        self.assertEqual(vocab.get_size(), 5)
        self.assertEqual(vocab.itos[vocab.get_size() - 1], "the")

        vocab = text_utils.VocabFromText(self.VOCAB_EXAMPLE_SENTENCES, min_count=11)
        self.assertEqual(vocab.get_size(), 4)

        vocab = text_utils.VocabFromText(
            self.VOCAB_EXAMPLE_SENTENCES, min_count=11, only_unk_extra=True
        )
        self.assertEqual(vocab.get_size(), 1)
        self.assertEqual(vocab.itos[vocab.get_size() - 1], "<unk>")

        vocab = text_utils.VocabFromText(
            self.VOCAB_EXAMPLE_SENTENCES, min_count=1, remove=[";"]
        )
        self.assertEqual(vocab.get_size(), 40)

        vocab = text_utils.VocabFromText(
            self.VOCAB_EXAMPLE_SENTENCES, min_count=1, remove=[";", ",", "?"]
        )
        self.assertEqual(vocab.get_size(), 38)

        vocab = text_utils.VocabFromText(
            self.VOCAB_EXAMPLE_SENTENCES, min_count=1, keep=["?"], remove=";"
        )
        self.assertEqual(vocab.get_size(), 40)

    def test_nucleus_sampling(self):
        vocab = text_utils.VocabFromText(self.VOCAB_EXAMPLE_SENTENCES)

        model_config = self.config.model_config.butd
        model = TestDecoderModel(model_config, vocab)
        model.build()
        model.to("cuda")
        model.eval()

        sample = Sample()
        sample.dataset_name = "coco"
        sample.dataset_type = "test"
        sample.image_feature_0 = torch.randn(100, 2048)
        sample.answers = torch.zeros((5, 10), dtype=torch.long)
        sample_list = SampleList([sample])

        tokens = model(sample_list)["captions"]

        # these are expected tokens for sum_threshold = 0.5
        expected_tokens = [
            1.0,
            6319.0,
            1516.0,
            3214.0,
            8798.0,
            4036.0,
            282.0,
            4706.0,
            8346.0,
            8620.0,
            2.0,
        ]

        self.assertEqual(tokens[0].tolist(), expected_tokens)
