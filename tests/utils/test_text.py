# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest

import mmf.utils.text as text_utils
import numpy as np
import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports
from mmf.utils.general import get_mmf_root
from packaging.version import LegacyVersion
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
        model.eval()

        sample = Sample()
        sample.dataset_name = "coco"
        sample.dataset_type = "test"
        sample.image_feature_0 = torch.randn(100, 2048)
        sample.answers = torch.zeros((5, 10), dtype=torch.long)
        sample_list = SampleList([sample])

        tokens = model(sample_list)["captions"]

        # these are expected tokens for sum_threshold = 0.5

        # Because of a bug fix in https://github.com/pytorch/pytorch/pull/47386
        # the torch.Tensor.multinomail will generate different random sequence.
        # TODO: Remove this hack after OSS uses later version of PyTorch.
        if LegacyVersion(torch.__version__) > LegacyVersion("1.7.1"):
            expected_tokens = [1.0, 23.0, 38.0, 30.0, 5.0, 11.0, 2.0]
        else:
            expected_tokens = [
                1.0,
                29.0,
                11.0,
                11.0,
                39.0,
                10.0,
                31.0,
                4.0,
                19.0,
                39.0,
                2.0,
            ]

        self.assertEqual(tokens[0].tolist(), expected_tokens)


class TestUtilsTextBeamSearch(unittest.TestCase):
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
            "beam_search.yaml",
        )
        config_path = os.path.abspath(config_path)
        args = dummy_args(model="butd", dataset="coco")
        args.opts.append(f"config={config_path}")
        configuration = Configuration(args)
        configuration.config.datasets = "coco"
        configuration.freeze()
        self.config = configuration.config
        registry.register("config", self.config)

    def test_beam_search(self):
        vocab = text_utils.VocabFromText(self.VOCAB_EXAMPLE_SENTENCES)
        model_config = self.config.model_config.butd
        model = TestDecoderModel(model_config, vocab)
        model.build()
        model.eval()

        expected_tokens = {
            1: [1.0, 23.0, 1.0, 24.0, 29.0, 37.0, 40.0, 17.0, 29.0, 2.0],
            2: [1.0, 0.0, 8.0, 1.0, 28.0, 25.0, 2.0],
            8: [1.0, 34.0, 1.0, 13.0, 1.0, 2.0],
            16: [1.0, 25.0, 18.0, 2.0],
        }

        for batch_size in [1, 2, 8, 16]:
            samples = []
            for _ in range(batch_size):
                sample = Sample()
                sample.dataset_name = "coco"
                sample.dataset_type = "test"
                sample.image_feature_0 = torch.randn(100, 2048)
                sample.answers = torch.zeros((5, 10), dtype=torch.long)
                samples.append(sample)

            sample_list = SampleList(samples)
            tokens = model(sample_list)["captions"]
            self.assertEqual(
                np.trim_zeros(tokens[0].tolist()), expected_tokens[batch_size]
            )
