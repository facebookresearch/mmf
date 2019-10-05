# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
import random
import os
import numpy as np

import pythia.utils.text_utils as text_utils

from tests.utils.test_model import TestDecoderModel
from pythia.common.registry import registry
from pythia.common.sample import Sample, SampleList
from pythia.utils.configuration import ConfigNode, Configuration
from pythia.utils.general import get_pythia_root


class TestTextUtils(unittest.TestCase):
    TOKENS = ["this", "will", "be", "a", "test", "of", "tokens"]
    TOKENIZE_EXAMPLE = "This will be a test of tokens?"
    VOCAB_EXAMPLE_SENTENCES = [
        "Are there more big green things than large purple shiny cubes?"
        "How many other things are there of the same shape as the tiny cyan matte object?",
        "Is the color of the large sphere the same as the large matte cube?"
        "What material is the big object that is right of the brown cylinder and "
        "left of the large brown sphere?",
        "How big is the brown shiny sphere? ;"
    ]

    def setUp(self):
        torch.manual_seed(1234)
        config_path = os.path.join(
            get_pythia_root(), "..", "configs", "captioning", "coco", "butd_nucleus_sampling.yml"
        )
        config_path = os.path.abspath(config_path)
        configuration = Configuration(config_path)
        configuration.config["datasets"] = "coco"
        configuration.config["model_attributes"]["butd"]["inference"]["params"]["sum_threshold"] = 0.5
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

        model_config = self.config.model_attributes.butd
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
        expected_tokens = [1.0000e+00, 2.9140e+03, 5.9210e+03, 2.2040e+03, 5.0550e+03, 9.2240e+03,
         4.5120e+03, 1.8200e+02, 3.6490e+03, 6.4090e+03, 2.0000e+00]

        self.assertEqual(tokens[0].tolist(), expected_tokens)

