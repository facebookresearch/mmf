# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import json
import pickle


def extract_qid_imid(ques_json_file):
    with open(ques_json_file, "r") as f:
        info = json.load(f)
        questions = info["questions"]

    q_im_ids = []
    for q in questions:
        im_id = q["image_id"]
        q_id = q["question_id"]
        q_im_ids.append((im_id, q_id))

    return q_im_ids


if __name__ == "__main__":
    minival_ques_file = "v2_OpenEnded_mscoco_minival2014_questions.json"

    val2train_ques_file = "v2_OpenEnded_mscoco_val2train2014_questions.json"

    minival_out_file = "data_prep/vqa_v2.0/minival_ids.pkl"
    val2train_out_file = "data_prep/vqa_v2.0/val2train_ids.pkl"

    minival_ids = extract_qid_imid(minival_ques_file)
    with open(minival_out_file, "wb") as w1:
        pickle.dump(minival_ids, w1)

    val2train_ids = extract_qid_imid(val2train_ques_file)
    with open(val2train_out_file, "wb") as w2:
        pickle.dump(val2train_ids, w2)
