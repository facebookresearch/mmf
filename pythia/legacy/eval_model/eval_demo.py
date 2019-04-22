# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import json
import sys

from eval_model.vqaEval import VQAEval


def parse_annotation(anno_file):
    with open(anno_file, "r") as f:
        annotations = json.load(f)["annotations"]

    q_2_anno = dict([(a["question_id"], a) for a in annotations])
    return q_2_anno


def parse_ans(answ_file):
    with open(answ_file, "r") as f:
        answers = json.load(f)

    q_2_answ = dict([(a["question_id"], a) for a in answers])
    return q_2_answ


if __name__ == "__main__":
    if len(sys.argv) < 3:
        exit(
            "USAGE: python eval_model/eval_demo.py \
             annotation_json_file answer_json_file"
        )

    anno_file = sys.argv[1]
    answ_file = sys.argv[2]

    q_2_anno = parse_annotation(anno_file)
    q_2_answ = parse_ans(answ_file)

    eval = VQAEval(q_2_anno, q_2_answ, 2)
    eval.evaluate()
    acc = eval.accuracy
    print(
        "overall: %.2f" % acc["overall"],
        "yes/no: %f" % acc["perAnswerType"]["yes/no"],
        "number: %.2f" % acc["perAnswerType"]["number"],
        "other: %.2f" % acc["perAnswerType"]["other"],
    )
