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

from dataset_utils.text_processing import tokenize

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

args = parser.parse_args()

input_files = args.input_files
out_dir = args.out_dir
min_freq = args.min_freq

os.makedirs(out_dir, exist_ok=True)

vocab_file_name = "vocabulary_vqa.txt"

word_count = Counter()
questions = []

for idx, input_file in enumerate(input_files):
    with open(input_file, "r") as f:
        questions += json.load(f)["questions"]

question_length = [None] * len(questions)

for inx, question in enumerate(questions):
    words = tokenize(question["question"])
    question_length[inx] = len(words)
    word_count.update(words)

vocabulary = [w[0] for w in word_count.items() if w[1] >= min_freq]
vocabulary.sort()
vocabulary = ["<unk>"] + vocabulary

vocab_file = os.path.join(out_dir, vocab_file_name)
with open(vocab_file, "w") as f:
    f.writelines([w + "\n" for w in vocabulary])


print("min question len=", min(question_length))
print("max question len=", max(question_length))
