# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import pythia.utils.text_utils as text_utils


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

