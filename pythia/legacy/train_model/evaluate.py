# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import os

import yaml
from torch.utils.data import DataLoader

from train_model.dataset_utils import prepare_eval_data_set
from train_model.Engineer import one_stage_eval_model
from train_model.eval_utils import get_final_validation
from train_model.model_factory import is_one_stageModel

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="config yaml file")
parser.add_argument("--out_dir", type=str, required=True, help="output directory")
args = parser.parse_args()

config_file = args.config
out_dir = args.out_dir

with open(config_file, "r") as f:
    config = yaml.load(f)

# get the potential shared data_config info
data_root_dir = config["data"]["data_root_dir"]
batch_size = config["data"]["batch_size"]
data_set_val = prepare_eval_data_set(**config["data"], **config["model"])
data_reader_val = DataLoader(data_set_val, shuffle=False, batch_size=batch_size)

snapshot_dir = os.path.join(out_dir, config["output"]["exp_name"])
os.makedirs(snapshot_dir, exist_ok=True)

model_type = config["model"]["model_type"]
if is_one_stageModel(model_type):
    get_final_validation(data_set_val, batch_size, snapshot_dir, one_stage_eval_model)
else:
    None
