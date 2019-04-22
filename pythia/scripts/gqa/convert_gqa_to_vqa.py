# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import os
import re
from collections import Counter

import h5py
import numpy as np
import tqdm

from pythia.utils.process_answers import preprocess_answer
from pythia.utils.text_processing import text_tokenize as tokenize


def merge_train(train_q_dir):
    merged_dic = {}

    for file_name in tqdm.tqdm(os.listdir(train_q_dir)):
        full_path = os.path.join(train_q_dir, file_name)
        partial_questions = json.load(open(full_path, "r"))
        merged_dic.update(partial_questions)

    save_dir = os.path.abspath(os.path.join(train_q_dir, os.pardir))

    with open(os.path.join(save_dir, "train_all_questions.json"), "w") as fp:
        json.dump(merged_dic, fp)


def get_objects(semantic_str):
    matches = re.findall("\(([^)]+)", semantic_str)
    result = []
    for match in matches:
        if "," in match:
            result += list(map(int, match.split(",")))
        elif match.isdigit():
            result += [int(match)]
        else:
            pass
    return result


def get_imdb(file_path):

    imdb = [{"dataset_name": "gqa"}]

    questions = json.load(open(file_path, "r"))
    print("Processing file {}".format(file_path))

    for qid, item in tqdm.tqdm(questions.items()):
        entry = {
            "image_name": item["imageId"] + "jpg",
            "image_id": item["imageId"],
            "question_id": qid,
            "question_str": item["question"],
            "question_tokens": tokenize(item["question"]),
        }

        if "answer" in item:
            entry["all_answers"] = [item["answer"] for _ in range(10)]
            entry["valid_answers"] = [item["answer"] for _ in range(10)]
            entry["semantic_string"] = (item["semanticStr"],)
            entry["gt_object_ids"] = (get_objects(item["semanticStr"]),)
            entry["meta_data"] = item["types"]

        imdb.append(entry)

    return np.array(imdb)


def extract_bbox_feats(feat_dir, out_dir):

    info_json_path = os.path.join(feat_dir, "gqa_objects_info.json")
    info_dict = json.load(open(info_json_path, "r"))

    file_mapping = {k: [] for k in range(16)}

    for k, v in info_dict.items():
        file_mapping[v["file"]] += [(k, v)]

    for i in range(16):
        file_path = os.path.join(feat_dir, "gqa_objects_{}.h5".format(i))
        print("Processing file {}".format(file_path))

        feat_db = h5py.File(file_path, "r")
        for entry in tqdm.tqdm(file_mapping[i]):
            image_id = entry[0]
            meta = entry[1]
            to_save = {
                "image_id": image_id,
                "boxes": feat_db["bboxes"][meta["idx"]],
                "feats": feat_db["features"][meta["idx"]],
                "height": meta["height"],
                "width": meta["width"],
                "n_objects": meta["objectsNum"],
            }

            save_path = os.path.join(out_dir, str(image_id) + ".npy")
            np.save(save_path, to_save)


def extract_spatial_feats(feat_dir, out_dir):
    info_json_path = os.path.join(feat_dir, "gqa_spatial_info.json")
    info_dict = json.load(open(info_json_path, "r"))

    file_mapping = {k: [] for k in range(16)}

    for k, v in info_dict.items():
        file_mapping[v["file"]] += [(k, v)]

    for i in range(16):
        file_path = os.path.join(feat_dir, "gqa_spatial_{}.h5".format(i))
        print("Processing file {}".format(file_path))

        feat_db = h5py.File(file_path, "r")
        for entry in tqdm.tqdm(file_mapping[i]):
            image_id = entry[0]
            meta = entry[1]
            to_save = feat_db["features"][meta["idx"]]
            to_save = to_save.reshape(1, 7, 7, 2048)
            save_path = os.path.join(out_dir, str(image_id) + ".npy")
            np.save(save_path, to_save)


def extract_image_features(image_dir, out_dir):
    extract_bbox_feats(
        os.path.join(image_dir, "objects"), os.path.join(out_dir, "objects")
    )

    extract_spatial_feats(
        os.path.join(image_dir, "spatial"), os.path.join(out_dir, "spatial")
    )


def convert_gqa_to_vqa(gqa_dir, out_dir):
    """
    Takes GQA dataset and converts it into VQA format

    Assumes GQA dir structure as:

    -gqa_dir/
      -images/
         -images/
         -objects/
         -spatial/
      -questions/
      -scenegraphs/
    """

    image_feat_path = os.path.join(gqa_dir, "images")
    extract_image_features(image_feat_path, out_dir)

    questions_dir = os.path.join(gqa_dir, "questions")

    if os.path.isfile(os.path.join(questions_dir, "train_all_questions.json")):
        print("Using previously generated train_all_questions.json file")
    else:
        merge_train(os.path.join(gqa_dir, "questions", "train_all_questions"))

    split_mapping = {
        "test": "test_all_questions.json",
        "val": "val_all_questions.json",
        "challenge": "challenge_all_questions.json",
        "train": "train_all_questions.json",
    }

    for split in split_mapping:
        for balance_type in ["balanced", "all"]:
            filename = split_mapping[split]
            csplit = split
            if balance_type == "balanced":
                filename = filename.replace("_all", "_balanced")
                csplit = split + "_balanced"

            file_path = os.path.join(questions_dir, filename)
            imdb = get_imdb(file_path)

            save_path = os.path.join(out_dir, "imdb_{}.npy".format(csplit))
            np.save(save_path, imdb)

    splits = ["val", "train"]
    split_type = ["balanced", "all"]

    global_answer = Counter()
    global_q = Counter()
    question_len = Counter()

    for s in splits:
        for st in split_type:
            questions_json = os.path.join(
                questions_dir, "{}_{}_questions.json".format(s, st)
            )
            questions = json.load(open(questions_json, "r"))

            print("Processing split {}_{}".format(s, st))

            answers = Counter()
            q_tokens = Counter()

            for qs, q in tqdm.tqdm(questions.items()):
                tokens = tokenize(q["question"])
                q_tokens.update(tokens)
                global_q.update(tokens)
                answers.update([q["answer"].lower()])
                global_answer.update([q["answer"].lower()])
                question_len.update([len(tokens)])

    print("N_unique answers :", len(global_answer))
    print("N unique q tokens:", len(global_q))
    print("Min Q length", min([x for x in question_len]))
    print("Max Q length", max([x for x in question_len]))
    print("Q length distribution", question_len)

    # Save question vocabulary
    q_vocabulary = [w[0] for w in global_q.items()]
    q_vocabulary.sort()
    q_vocabulary = ["<unk>"] + q_vocabulary

    vocab_file = os.path.join(out_dir, "vocabulary_gqa.txt")
    with open(vocab_file, "w") as f:
        f.writelines([w + "\n" for w in q_vocabulary])

    # Save answer vocabulary
    answer_list = [preprocess_answer(ans[0]) for ans in global_answer.items()]
    answer_list = [t.strip() for t in answer_list if len(t.strip()) > 0]
    answer_list.sort()

    if "<unk>" not in answer_list:
        answer_list = ["<unk>"] + answer_list

    answer_file = os.path.join(out_dir, "answers_gqa.txt")
    with open(answer_file, "w") as fp:
        fp.writelines([w + "\n" for w in answer_list])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    convert_gqa_to_vqa(args.gqa_dir, args.out_dir)
