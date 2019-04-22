# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch

imdb_version = 1
use_cuda = torch.cuda.is_available()

model_type_gt = "gt_layout"
model_type_scratch = "scratch"
model_type_gt_rl = "gt+rl"
model_type_top_down_bottom_up = "top_down_bottom_up"


topdown_concate_attention = "concate_attention"
topdown_project_attention = "project_attention"
