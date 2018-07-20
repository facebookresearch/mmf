# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np


def unique_columns(data):
    dt = np.dtype((np.void, data.dtype.itemsize * data.shape[0]))
    dataf = np.asfortranarray(data).view(dt)
    u, uind = np.unique(dataf, return_inverse=True)
    m = u.view(data.dtype).reshape(-1, data.shape[0]).T
    res = [np.where(uind == x)[0] for x in range(m.shape[1])]
    return res
