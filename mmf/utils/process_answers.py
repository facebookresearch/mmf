# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import os

from mmf.datasets.processors.processors import EvalAIAnswerProcessor


def get_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1


def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


def filter_answers(answers_dset, min_occurence):
    """This will change the answer to preprocessed version
    """
    occurence = {}
    answer_list = []
    evalai_answer_processor = EvalAIAnswerProcessor()
    for ans_entry in answers_dset:
        gtruth = ans_entry["multiple_choice_answer"]
        gtruth = evalai_answer_processor(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry["question_id"])
    for answer in occurence.keys():
        if len(occurence[answer]) >= min_occurence:
            answer_list.append(answer)

    print(
        "Num of answers that appear >= %d times: %d" % (min_occurence, len(answer_list))
    )
    return answer_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="input train annotationjson file",
    )
    parser.add_argument(
        "--val_annotation_file",
        type=str,
        required=False,
        help="input val annotation json file",
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
        help="the minimum times of answer occurrence \
                              to be included in vocabulary, default 0",
    )
    args = parser.parse_args()

    train_annotation_file = args.annotation_file
    out_dir = args.out_dir
    min_freq = args.min_freq

    answer_file_name = "answers_vqa.txt"
    os.makedirs(out_dir, exist_ok=True)

    train_answers = json.load(open(train_annotation_file))["annotations"]
    answers = train_answers

    if args.val_annotation_file is not None:
        val_annotation_file = args.val_annotation_file
        val_answers = json.load(open(val_annotation_file))["annotations"]
        answers = train_answers + val_answers

    answer_list = filter_answers(answers, min_freq)
    answer_list = [t.strip() for t in answer_list if len(t.strip()) > 0]
    answer_list.sort()

    if "<unk>" not in answer_list:
        answer_list = ["<unk>"] + answer_list

    answer_file = os.path.join(out_dir, answer_file_name)
    with open(answer_file, "w") as f:
        f.writelines([w + "\n" for w in answer_list])
