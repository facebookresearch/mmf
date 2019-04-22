# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import json
import os

import numpy as np

from dataset_utils import text_processing
from dataset_utils.create_imdb_header import create_header


def extract_answers(q_answers, valid_answer_set):
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    return all_answers, valid_answers


def build_imdb(
    image_set, valid_answer_set, coco_set_name=None, annotation_set_name=None
):
    annotation_file = os.path.join(data_dir, "v2_mscoco_%s_annotations.json")
    question_file = os.path.join(data_dir, "v2_OpenEnded_mscoco_%s_questions.json")

    print("building imdb %s" % image_set)
    has_answer = False
    has_gt_layout = False
    load_gt_layout = False
    load_answer = False

    annotation_set_name = (
        annotation_set_name if annotation_set_name is not None else image_set
    )

    if os.path.exists(annotation_file % annotation_set_name):
        with open(annotation_file % annotation_set_name) as f:
            annotations = json.load(f)["annotations"]
            qid2ann_dict = {ann["question_id"]: ann for ann in annotations}
        load_answer = True
    """
    if image_set in ['train2014', 'val2014']:
        load_answer = True
        load_gt_layout = False
        with open(annotation_file % image_set) as f:
            annotations = json.load(f)["annotations"]
            qid2ann_dict = {ann['question_id']: ann for ann in annotations}
        #qid2layout_dict = np.load(gt_layout_file % image_set)[()]
    else:
        load_answer = False
        load_gt_layout = False """

    with open(question_file % image_set) as f:
        questions = json.load(f)["questions"]
    coco_set_name = (
        coco_set_name if coco_set_name is not None else image_set.replace("-dev", "")
    )
    image_name_template = "COCO_" + coco_set_name + "_%012d"
    imdb = [None] * (len(questions) + 1)

    unk_ans_count = 0
    for n_q, q in enumerate(questions):
        if (n_q + 1) % 10000 == 0:
            print("processing %d / %d" % (n_q + 1, len(questions)))
        image_id = q["image_id"]
        question_id = q["question_id"]
        image_name = image_name_template % image_id
        feature_path = image_name + ".npy"
        question_str = q["question"]
        question_tokens = text_processing.tokenize(question_str)

        iminfo = dict(
            image_name=image_name,
            image_id=image_id,
            question_id=question_id,
            feature_path=feature_path,
            question_str=question_str,
            question_tokens=question_tokens,
        )

        # load answers
        if load_answer:
            ann = qid2ann_dict[question_id]
            all_answers, valid_answers = extract_answers(
                ann["answers"], valid_answer_set
            )
            if len(valid_answers) == 0:
                valid_answers = ["<unk>"]
                unk_ans_count += 1
            iminfo["all_answers"] = all_answers
            iminfo["valid_answers"] = valid_answers
            has_answer = True

        if load_gt_layout:
            has_gt_layout = True

        imdb[n_q + 1] = iminfo
    print("total %d out of %d answers are <unk>" % (unk_ans_count, len(questions)))
    header = create_header("vqa", has_answer=has_answer, has_gt_layout=has_gt_layout)
    imdb[0] = header
    return imdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="data directory")
    parser.add_argument(
        "--out_dir", type=str, required=True, help="imdb output directory"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir

    vocab_answer_file = os.path.join(out_dir, "answers_vqa.txt")
    answer_dict = text_processing.VocabDict(vocab_answer_file)
    valid_answer_set = set(answer_dict.word_list)

    imdb_train2014 = build_imdb("train2014", valid_answer_set)
    imdb_val2014 = build_imdb("val2014", valid_answer_set)
    imdb_test2015 = build_imdb("test2015", valid_answer_set)

    imdb_dir = os.path.join(out_dir, "imdb")
    os.makedirs(imdb_dir, exist_ok=True)
    np.save(os.path.join(imdb_dir, "imdb_train2014.npy"), np.array(imdb_train2014))
    np.save(os.path.join(imdb_dir, "imdb_val2014.npy"), np.array(imdb_val2014))
    np.save(os.path.join(imdb_dir, "imdb_test2015.npy"), np.array(imdb_test2015))

    imdb_minival2014 = build_imdb(
        "minival2014",
        valid_answer_set,
        coco_set_name="val2014",
        annotation_set_name="val2014",
    )
    imdb_val2train2014 = build_imdb(
        "val2train2014",
        valid_answer_set,
        coco_set_name="val2014",
        annotation_set_name="val2014",
    )

    np.save(os.path.join(imdb_dir, "imdb_minival2014.npy"), np.array(imdb_minival2014))
    np.save(
        os.path.join(imdb_dir, "imdb_val2train2014.npy"), np.array(imdb_val2train2014)
    )
