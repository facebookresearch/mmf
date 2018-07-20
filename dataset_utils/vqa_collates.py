# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from torch.utils.data.dataloader import default_collate
import numpy as np


def filter_unk_collate(batch):
    batch = list(filter(lambda x: np.sum(x['ans_scores']) > 0, batch))
    return default_collate(batch)
