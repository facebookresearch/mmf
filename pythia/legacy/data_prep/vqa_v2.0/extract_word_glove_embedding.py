# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import os

import numpy as np

from dataset_utils.text_processing import VocabDict


def subset_weights(glove_file, vocabulary_file):
    with open(glove_file, "r") as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(" ")) - 1
    print("embedding dim is %d" % emb_dim)

    vocabulary = VocabDict(vocab_file=vocabulary_file)

    weights = np.zeros((vocabulary.num_vocab, emb_dim), dtype=np.float32)

    word2emb = {}
    for entry in entries:
        vals = entry.split(" ")
        word = vals[0]
        vals = np.array(list(map(float, vals[1:])))
        word2emb[word] = np.array(vals)

    for word, idx in vocabulary.word2idx_dict.items():
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]

    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocabulary_file",
        type=str,
        required=True,
        help="input train annotationjson file",
    )
    parser.add_argument(
        "--glove_file",
        type=str,
        required=True,
        help="glove files with the corresponding dim",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./",
        help="output directory, default is current directory",
    )

    args = parser.parse_args()

    glove_file = args.glove_file
    vocabulary_file = args.vocabulary_file
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    emb_file_name = "vqa2.0_" + os.path.basename(glove_file) + ".npy"

    weights = subset_weights(glove_file, vocabulary_file)

    emb_file = os.path.join(out_dir, emb_file_name)
    np.save(emb_file, weights)
