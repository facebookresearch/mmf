# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import json
import os
from collections import Counter

from pythia.utils.text_utils import tokenize


class ExtractVocabulary:
    def __init__(self):
        self.args = self.get_args()
        self.input_files = self.args.input_files
        self.out_dir = self.args.out_dir
        self.min_freq = self.args.min_freq
        self.vocab_file_name = self.args.vocab_file_name

    def extract(self):
        os.makedirs(self.out_dir, exist_ok=True)

        word_count = Counter()

        texts = self.get_text()
        text_lengths = [None] * len(texts)

        for inx, text in enumerate(texts):
            words = tokenize(text)
            text_lengths[inx] = len(words)
            word_count.update(words)

        # UNK token will added on fly if you use Vocab class in core/text
        vocabulary = [w[0] for w in word_count.items() if w[1] >= self.min_freq]
        vocabulary.sort()

        self.save_vocabulary(vocabulary)

        print("min text len=", min(text_lengths))
        print("max text len=", max(text_lengths))

    def save_vocabulary(self, vocabulary):
        vocab_file = os.path.join(self.out_dir, self.vocab_file_name)
        with open(vocab_file, "w") as f:
            f.writelines([w + "\n" for w in vocabulary])

    def get_text(self):
        """
        Override this in your child class to extract custom text
        Default for VQA. Make sure to return a list of all possible text
        """
        text = []

        for input_file in self.input_files:
            with open(input_file, "r") as f:
                text += json.load(f)["questions"]

        return text

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input_files",
            nargs="+",
            required=True,
            help="input question json files, \
                                 if more than 1, split by space",
        )
        parser.add_argument(
            "--out_dir",
            type=str,
            default="./",
            help="output directory, default is current directory",
        )
        parser.add_argument(
            "--min_freq",
            type=int,
            default=0,
            help="the minimum times of word occurrence \
                                  to be included in vocabulary, default 0",
        )
        parser.add_argument(
            "--vocab_file_name",
            type=str,
            default="vocabulary.txt",
            help="Name of the file in vocabulary will be stored",
        )

        args = parser.parse_args()

        return args


if __name__ == "__main__":
    extractor = ExtractVocabulary()
    extractor.extract()
