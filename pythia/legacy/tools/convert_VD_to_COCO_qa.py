# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import json
import string

# The paths need to be updated
visdial_train_data_file = "visdial_0.9_train.json"
visdial_val_data_file = "visdial_0.9_val.json"


vdtrain_questions_file = "v2_OpenEnded_mscoco_vdtrain_questions.json"
vdval_questions_file = "v2_OpenEnded_mscoco_vdval_questions.json"

vdtrain_annotations_file = "v2_mscoco_vdtrain_annotations.json"
vdval_annotations_file = "v2_mscoco_vdval_annotations.json"


translator = str.maketrans("", "", string.punctuation)

vdtrain_questions = []
vdval_questions = []
vdtrain_annotations = []
vdval_annotations = []

with open(visdial_train_data_file, "r") as f:
    vdtrain_data = json.load(f)
    vdtrain_data_questions = vdtrain_data["data"]["questions"]
    vdtrain_data_answers = vdtrain_data["data"]["answers"]
    vdtrain_dialogs = vdtrain_data["data"]["dialogs"]
    count = 1
    for dialogs in vdtrain_data["data"]["dialogs"]:
        image_id = dialogs["image_id"]
        for dialog in dialogs["dialog"]:
            qid = dialog["question"]
            aid = dialog["answer"]
            q = vdtrain_data_questions[qid]
            a = vdtrain_data_answers[aid]
            question = {}
            annotation = {}
            question["image_id"] = image_id
            question["question_id"] = count
            question["question"] = q
            vdtrain_questions.append(question)
            a = a.translate(translator)
            a = a.lower()
            annotation["multiple_choice_answer"] = a
            annotation["question_id"] = count
            annotation["answers"] = []
            for i in range(10):
                answer = {}
                answer["answer"] = a
                answer["answer_confifence"] = "yes"
                answer["answer_id"] = i + 1
                annotation["answers"].append(answer)
            vdtrain_annotations.append(annotation)
            count = count + 1
    print("Total qa train " + str(count))


with open(visdial_val_data_file, "r") as f:
    vdval_data = json.load(f)
    vdval_data_questions = vdval_data["data"]["questions"]
    vdval_data_answers = vdval_data["data"]["answers"]
    vdval_dialogs = vdval_data["data"]["dialogs"]
    count = 1
    for dialogs in vdval_data["data"]["dialogs"]:
        image_id = dialogs["image_id"]
        for dialog in dialogs["dialog"]:
            qid = dialog["question"]
            aid = dialog["answer"]
            q = vdtrain_data_questions[qid]
            a = vdtrain_data_answers[aid]
            question = {}
            annotation = {}
            question["image_id"] = image_id
            question["question_id"] = count
            question["question"] = q
            vdval_questions.append(question)
            a = a.lower()
            a = a.translate(translator)
            annotation["multiple_choice_answer"] = a
            annotation["question_id"] = count
            annotation["answers"] = []
            for i in range(10):
                answer = {}
                answer["answer"] = a
                answer["answer_confifence"] = "yes"
                answer["answer_id"] = i + 1
                annotation["answers"].append(answer)
            vdval_annotations.append(annotation)
            count = count + 1
    print("Total qa val " + str(count))

vdtrain_data = {}
vdtrain_data["questions"] = vdtrain_questions

vdval_data = {}
vdval_data["questions"] = vdval_questions

with open(vdtrain_questions_file, "w") as f:
    json.dump(vdtrain_data, f)

with open(vdval_questions_file, "w") as f:
    json.dump(vdval_data, f)

vdtrain_data = {}
vdtrain_data["annotations"] = vdtrain_annotations

vdval_data = {}
vdval_data["annotations"] = vdval_annotations


with open(vdtrain_annotations_file, "w") as f:
    json.dump(vdtrain_data, f)

with open(vdval_annotations_file, "w") as f:
    json.dump(vdval_data, f)
