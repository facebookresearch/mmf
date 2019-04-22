# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os

import torch
from torch.utils.data import DataLoader


def get_final_validation(data_set_val, batch_size, snapshot_dir, eval_model):
    final_val_data_reader = DataLoader(
        data_set_val, shuffle=False, batch_size=batch_size
    )

    files = [
        os.path.join(snapshot_dir, file)
        for file in os.listdir(snapshot_dir)
        if file.startswith("model")
    ]

    for model_file in sorted(files, key=os.path.getctime, reverse=True):
        current_model = torch.load(model_file)
        total_sample = 0
        total_score = 0
        for i, batch in enumerate(final_val_data_reader):
            score, n_sample, _ = eval_model(batch, current_model)
            total_sample += n_sample
            total_score += score

        acc = total_score / total_sample
        print(model_file, ": %.6f" % acc)
