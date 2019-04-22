# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import json
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ques_file", type=str)
    pass


if __name__ == "__main__":
    val_json_file = "v2_OpenEnded_mscoco_val2014_questions.json"
    minival_json_file = "v2_OpenEnded_mscoco_minival2014_questions.json"
    val_as_train_json_file = "v2_OpenEnded_mscoco_val2train2014_questions.json"

    with open(val_json_file, "r") as f:
        file_info = json.load(f)
        questions = file_info["questions"]
        info = file_info["info"]
        task_type = file_info["task_type"]
        data_type = file_info["data_type"]
        license = file_info["license"]
        data_subtype = file_info["info"]

    # collect image_id
    image_ids = []
    for q in questions:
        image_id = q["image_id"]
        image_ids.append(image_id)

    # divide image_ids to two parts
    random.shuffle(image_ids)
    minival_images = image_ids[:10000]
    other_images = image_ids[10000:]

    minival_ques = []
    other_ques = []

    total_minival = 0
    total_others = 0
    # seprate quesion_json_file
    for q in questions:
        image_id = q["image_id"]

        if image_id in minival_images:
            minival_ques.append(q)
            total_minival += 1
        else:
            other_ques.append(q)
            total_others += 1

    minival_json = {
        "info": info,
        "task_type": task_type,
        "data_type": data_type,
        "license": license,
        "data_subtype": "minival2014",
        "questions": minival_ques,
    }

    other_json = {
        "info": info,
        "task_type": task_type,
        "data_type": data_type,
        "license": license,
        "data_subtype": "val2train2014",
        "questions": other_ques,
    }

    with open(minival_json_file, "w") as w1:
        json.dump(minival_json, w1)

    with open(val_as_train_json_file, "w") as w2:
        json.dump(other_json, w2)

    print(
        "minival_questions: %d" % total_minival + "other_questions: %d" % total_others
    )
