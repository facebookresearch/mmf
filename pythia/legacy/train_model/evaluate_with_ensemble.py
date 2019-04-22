# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import glob
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import _pickle as pickle
from train_model.dataset_utils import prepare_eval_data_set
from train_model.Engineer import masked_unk_softmax, one_stage_run_model
from train_model.model_factory import is_one_stageModel

tmp_model_file_name = "tmp_result_%d.pkl"
tmp_model_file_name_pattern = "tmp_result*.pkl"


class answer_json:
    def __init__(self):
        self.answers = []

    def add(self, ques_id, ans):
        res = {"question_id": ques_id, "answer": ans}
        self.answers.append(res)


def compute_score_with_prob(prob, scores):
    max_prob_pos = prob.max(axis=1, keepdims=1) == prob
    score_sum = np.sum(scores * max_prob_pos)
    return score_sum


def ensemble(results, ans_unk_idx):
    final_result = masked_unk_softmax(results[0], dim=1, mask_idx=ans_unk_idx)

    if len(results) == 1:
        return final_result

    for result in results[1:]:
        final_result += masked_unk_softmax(result, dim=1, mask_idx=ans_unk_idx)

    return final_result


def ensemble_model(model_dir, max_model=None, clear=True):
    count = 0
    final_result = None
    for model_file in glob.glob(os.path.join(model_dir, tmp_model_file_name_pattern)):
        count += 1
        if max_model is not None and count > max_model:
            break
        with open(os.path.join(model_dir, model_file), "rb") as f:
            pred_result = pickle.load(f)
        if final_result is None:
            final_result = pred_result
        else:
            final_result += pred_result

        # remove tmp file after ensembling
        if clear:
            os.remove(os.path.join(model_dir, model_file))

    return final_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config yaml file")
    parser.add_argument("--out_dir", type=str, required=True, help="output dir")
    parser.add_argument(
        "--model_paths", nargs="+", help="paths for model", default=None
    )
    parser.add_argument(
        "--model_dirs", nargs="+", help="directories for models", default=None
    )
    args = parser.parse_args()

    config_file = args.config
    out_dir = args.out_dir
    model_files = args.model_paths
    model_dirs = args.model_dirs

    with open(config_file, "r") as f:
        config = yaml.load(f)

    # get the potential shared data_config info
    data_root_dir = config["data"]["data_root_dir"]
    batch_size = config["data"]["batch_size"]
    data_set_val = prepare_eval_data_set(
        **config["data"], **config["model"], verbose=True
    )
    data_reader_val = DataLoader(data_set_val, shuffle=True, batch_size=100)

    ans_dic = data_set_val.answer_dict

    ans_json_out = answer_json()

    current_models = (
        []
        if model_files is None
        else [torch.load(model_file) for model_file in model_files]
    )

    if model_dirs is not None:
        for model_dir in model_dirs:
            for file in glob.glob(model_dir + "/**/best_model.pth", recursive=True):
                this_model = torch.load(file)
                current_models.append(this_model)

    if len(current_models) == 0:
        exit("no model provided")

    model_type = config["model"]["model_type"]

    total_score = 0
    total_max_score = 0
    total_sample = 0

    num_of_model = len(current_models)
    os.makedirs(out_dir, exist_ok=True)

    if is_one_stageModel(model_type):
        for i, batch in enumerate(data_reader_val):
            if i % 100 == 0:
                print("process batch %d" % i)
            verbose_info = batch["verbose_info"]
            answer_scores = batch["ans_scores"]
            answer_scores_np = answer_scores.numpy()

            for imd, current_model in enumerate(current_models):
                # print("process model %d"%imd)
                logit_res = one_stage_run_model(batch, current_model)

                softmax_res = masked_unk_softmax(
                    logit_res, dim=1, mask_idx=ans_dic.UNK_idx
                )
                softmax_res_data = softmax_res.data.cpu().numpy()

                with open(os.path.join(out_dir, tmp_model_file_name % imd), "wb") as w:
                    pickle.dump(softmax_res_data, w)

            ensembled_soft_max_result = ensemble_model(out_dir, num_of_model)

            nsample, _ = answer_scores_np.shape
            total_sample += nsample
            scores = compute_score_with_prob(
                ensembled_soft_max_result, answer_scores_np
            )
            total_score += scores

        print(
            "model: %d, sample= %d, score =%.6f"
            % (num_of_model, total_sample, total_score / total_sample)
        )
