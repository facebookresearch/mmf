# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from pythia.utils.preprocessing import text_tokenize


class TestUtilsPreprocessing(unittest.TestCase):
    TOKENS = ["this", "will", "be", "a", "test", "of", "tokens"]
    SENTENCE = "This will be a test of tokens?"

    def test_text_tokenize(self):
        tokens = text_tokenize(self.SENTENCE)

        self.assertEqual(list(tokens), self.TOKENS)
