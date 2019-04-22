# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import glob
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from train_model.dataset_utils import prepare_eval_data_set
from train_model.helper import build_model, run_model

CONFIG = "config.yaml"
MODELNAME = "best_model.pth"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit(
            "USAGE: python tools/eval_ensemble_on_val.py parent_dir \
             [ensemble sizes]"
        )

    esbl_sizes = [int(a) for a in sys.argv[2:]]

    parent_dir = sys.argv[1]

    model_pths = [
        file for file in glob.glob(parent_dir + "/**/" + MODELNAME, recursive=True)
    ]
    config_files = [c.replace(MODELNAME, CONFIG) for c in model_pths]

    if len(esbl_sizes) == 0:
        esbl_sizes = range(1, len(config_files) + 1)

    config_file = config_files[0]

    with open(config_file, "r") as f:
        config = yaml.load(f)

    batch_size = config["data"]["batch_size"]
    data_set_test = prepare_eval_data_set(
        **config["data"], **config["model"], verbose=True
    )
    data_reader_test = DataLoader(
        data_set_test, shuffle=False, batch_size=batch_size, num_workers=5
    )
    ans_dic = data_set_test.answer_dict

    accumulated_softmax = None
    final_result = {}
    n_model = 0
    for c_file, model_file in zip(config_files, model_pths):
        with open(c_file, "r") as f:
            config = yaml.load(f)

        myModel = build_model(config, data_set_test)
        myModel.load_state_dict(torch.load(model_file)["state_dict"])

        question_ids, soft_max_result = run_model(
            myModel, data_reader_test, ans_dic.UNK_idx
        )

        if n_model == 0:
            final_result = soft_max_result
