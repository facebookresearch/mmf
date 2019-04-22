# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import glob
import json

import numpy as np

import _pickle as pickle
from train_model.helper import print_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="output file name")
    parser.add_argument(
        "--res_dirs",
        nargs="+",
        help="directories for results, NOTE:"
        "all *.pkl file under these dirs will be ensembled",
        default=None,
    )
    argments = parser.parse_args()

    return argments


class answer_json:
    def __init__(self):
        self.answers = []

    def add(self, ques_id, ans):
        res = {"question_id": ques_id, "answer": ans}
        self.answers.append(res)


if __name__ == "__main__":

    args = parse_args()
    result_dirs = args.res_dirs
    out_file = args.out
    question_ids = None
    soft_max_result = None
    ans_dic = None
    cnt = 0
    for res_dir in result_dirs:
        for file in glob.glob(res_dir + "/**/*.pkl", recursive=True):
            with open(file, "rb") as f:
                cnt += 1
                sm = pickle.load(f)
                if soft_max_result is None:
                    soft_max_result = sm
                    question_ids = pickle.load(f)
                    ans_dic = pickle.load(f)
                else:
                    soft_max_result += sm

    print("ensemble total %d models" % cnt)

    predicted_answers = np.argmax(soft_max_result, axis=1)

    pkl_file = out_file + ".pkl"

    print_result(question_ids, soft_max_result, ans_dic, out_file, False, pkl_file)

    print("Done")
