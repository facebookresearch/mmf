# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse

import torch
import torch.nn.functional as F
import yaml
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from global_variables.global_variables import use_cuda
from main import my_collate
from train_model.dataset_utils import prepare_eval_data_set
from train_model.model_factory import is_one_stageModel


def one_stage_model_error_analysis(batch, myModel, answer_dict, writer):
    n_sample, _ = batch["input_seq_batch"].shape
    input_text_seq_lens = batch["seq_length_batch"]
    input_text_seqs = batch["input_seq_batch"]
    input_images = batch["image_feat_batch"]

    input_valid_answers = batch["valid_ans_label_batch"]

    input_txt_variable = Variable(input_text_seqs.type(torch.LongTensor))
    input_txt_variable = input_txt_variable.cuda() if use_cuda else input_txt_variable

    image_feat_variable = Variable(input_images)
    image_feat_variable = (
        image_feat_variable.cuda() if use_cuda else image_feat_variable
    )

    logit_res = myModel(
        input_question_variable=input_txt_variable,
        input_text_seq_lens=input_text_seq_lens,
        image_feat_variable=image_feat_variable,
    )

    predicted_answers = torch.topk(logit_res, 1)[1].cpu().data.numpy()[:, 0]
    predicted_answers_prob = torch.max(F.softmax(logit_res, dim=1), dim=1)[0]

    score = 0
    verbose_info = batch["verbose_info"]

    for idx, valid_answer_variable in enumerate(input_valid_answers):

        pred_ans = predicted_answers[idx]
        pred_ans_prob = predicted_answers_prob[idx]
        pred_ans_des = answer_dict.idx2word(pred_ans)
        valid_answer = valid_answer_variable.cpu().tolist()
        pred_ans_count = valid_answer.count(pred_ans)
        score = min(pred_ans_count * 0.3, 1)
        image_name = verbose_info["image_name"][idx]
        question_id = str(verbose_info["question_id"][idx])
        question = verbose_info["question_str"][idx]

        writer.write(
            question_id
            + "\t"
            + image_name
            + "\t"
            + '"'
            + question
            + '"'
            + "\t"
            + '"'
            + pred_ans_des
            + '"'
            + "\t"
            + str(pred_ans_count)
            + "\t"
            + str(score)
            + "\t"
            + "%.5f" % pred_ans_prob
            + "\n"
        )

    return score, n_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config yaml file")
    parser.add_argument("--out_file", type=str, required=True, help="output file")
    parser.add_argument(
        "--model_path", type=str, required=True, help="file path to model"
    )
    args = parser.parse_args()

    config_file = args.config

    with open(config_file, "r") as f:
        config = yaml.load(f)

    out_writer = open(args.out_file, "w+")

    # get the potential shared data_config info
    data_root_dir = config["data"]["data_root_dir"]
    batch_size = config["data"]["batch_size"]
    data_set_val = prepare_eval_data_set(
        **config["data"], **config["model"], verbose=True
    )

    collate_fun = default_collate
    if "adaptive_location" in config["data"] and config["data"]["adaptive_location"]:
        collate_fun = my_collate

    data_reader_val = DataLoader(
        data_set_val, shuffle=False, batch_size=batch_size, collate_fn=collate_fun
    )
    myModel = torch.load(args.model_path)

    model_type = config["model"]["model_type"]
    if is_one_stageModel(model_type):
        for i, batch_data in enumerate(data_reader_val):
            one_stage_model_error_analysis(
                batch_data, myModel, data_set_val.answer_dict, out_writer
            )
    else:
        None

    out_writer.close()
