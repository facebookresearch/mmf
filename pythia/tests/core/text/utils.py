# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import pythia.common.text.utils as text_utils


class TestTextUtils(unittest.TestCase):
    TOKENS = ["this", "will", "be", "a", "test", "of", "tokens"]

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
