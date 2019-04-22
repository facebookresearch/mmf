# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import pickle
import sys

if len(sys.argv) < 4:
    exit(
        "USAGE: python tools/extract_detectron_weights.py \
         weights_file out_dir feat_name [feat_name]"
    )

wgts_file = sys.argv[1]
out_dir = sys.argv[2]

with open(wgts_file, "rb") as f:
    wgts = pickle.load(f, encoding="latin1")["blobs"]

for i in range(3, len(sys.argv)):
    feat_name = sys.argv[i]
    wgt = wgts[feat_name]
    out_file = os.path.join(out_dir, feat_name + ".pkl")
    with open(out_file, "wb") as w:
        pickle.dump(wgt, w)
