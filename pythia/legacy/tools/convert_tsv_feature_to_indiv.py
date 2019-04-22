# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import base64
import csv
import os
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, required=True, help="input file")
parser.add_argument("--label", type=str, required=True, help="label for dataset")
parser.add_argument("--out_dir", type=str, required=True, help="imdb output directory")
args = parser.parse_args()

out_dir = args.out_dir


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features"]
infile = args.infile

label = args.label

out_dir = os.path.join(out_dir, label)

os.makedirs(out_dir, exist_ok=True)

print("reading tsv...")
with open(infile, "r") as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES)
    for item in reader:
        item["num_boxes"] = int(item["num_boxes"])
        image_id = int(item["image_id"])
        image_w = float(item["image_w"])
        image_h = float(item["image_h"])

        image_bboxes = np.frombuffer(
            base64.b64decode(item["boxes"]), dtype=np.float32
        ).reshape((item["num_boxes"], -1))

        image_feat = np.frombuffer(
            base64.b64decode(item["features"]), dtype=np.float32
        ).reshape((item["num_boxes"], -1))

        image_feat_and_boxes = {"image_bboxes": image_bboxes, "image_feat": image_feat}

        image_file_name = os.path.join(
            out_dir, "COCO_" + label + "_%012d.npy" % image_id
        )
        np.save(image_file_name, image_feat_and_boxes)
