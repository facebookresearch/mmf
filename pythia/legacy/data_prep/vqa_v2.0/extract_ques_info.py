# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import json


def extract_info(annotations, writer):
    for annotation in annotations:
        question_id = annotation["question_id"]
        answer_type = annotation["answer_type"]
        question_type = annotation["question_type"]
        multiple_choice_answer = annotation["multiple_choice_answer"]
        answers = [a["answer"] for a in annotation["answers"]]
        answers_out = "|".join([str(a) for a in answers])
        confidences = [a["answer_confidence"] for a in annotation["answers"]]
        confidences_out = "|".join(str(a) for a in confidences)

        writer.write(
            str(question_id)
            + "\t"
            + question_type
            + "\t"
            + answer_type
            + "\t"
            + str(multiple_choice_answer)
            + "\t"
            + answers_out
            + "\t"
            + confidences_out
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_files",
        nargs="+",
        required=True,
        help="input annotation json files, \
                             if more than 1, split by space",
    )
    parser.add_argument("--out", type=str, required=True, help="out put files")

    args = parser.parse_args()
    out_writer = open(args.out, "w")

    for annotation_file in args.annotation_files:
        with open(annotation_file, "r") as f:
            annotations = json.load(f)["annotations"]
        extract_info(annotations, out_writer)

    out_writer.close()
