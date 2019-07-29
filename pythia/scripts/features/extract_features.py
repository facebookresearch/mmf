#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""
# Category mapping for visual genome can be downloaded from
# https://dl.fbaipublicfiles.com/pythia/data/visual_genome_categories.json
# When the --background flag is set, the index saved with key "objects" in
# info_list will be +1 of the Visual Genome category mapping above and 0
# is the background class. When the --background flag is not set, the
# index saved with key "objects" in info list will match the Visual Genome
# category mapping.

from __future__ import absolute_import, division, print_function

import argparse
import base64
import csv
import glob
import json
import logging
import os
import sys
import timeit
from collections import defaultdict

import caffe2
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
from caffe2.python import workspace

import common.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
from common.config import assert_and_infer_cfg, cfg, merge_cfg_from_file
from utils.boxes import nms
from utils.io import cache_url
from utils.timer import Timer

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

csv.field_size_limit(sys.maxsize)

BOTTOM_UP_FIELDNAMES = [
    "image_id",
    "image_w",
    "image_h",
    "num_boxes",
    "boxes",
    "features",
]


FIELDNAMES = [
    "image_id",
    "image_w",
    "image_h",
    "num_boxes",
    "boxes",
    "features",
    "object",
]


# Run using
# extract_detectron_feat.py --cfg ${d_23_cfg}  --image-ext jpg  --wts ${d_23_md} --output_dir ${out_dir}/test2015  --feat_name gpu_0/fc6  --min_bboxes 100 --max_bboxes 100  ${input_dir}


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end inference")
    parser.add_argument(
        "--cfg",
        dest="cfg",
        help="cfg model file (/path/to/model_config.yaml)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--wts",
        dest="weights",
        help="weights model file (/path/to/model_weights.pkl)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="output dir name",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--image-ext",
        dest="image_ext",
        help="image file name extension (default: jpg)",
        default="jpg",
        type=str,
    )
    parser.add_argument(
        "--bbox_file", help="csv file from bottom-up attention model", default=None
    )
    parser.add_argument(
        "--total_group", help="the number of group for exracting", type=int, default=1
    )
    parser.add_argument(
        "--group_id",
        help=" group id for current analysis, used to shard",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--min_bboxes", help=" min number of bboxes", type=int, default=10
    )
    parser.add_argument(
        "--max_bboxes", help=" min number of bboxes", type=int, default=100
    )
    parser.add_argument(
        "--feat_name",
        help=" the name of the feature to extract, default: gpu_0/fc7",
        type=str,
        default="gpu_0/fc7",
    )
    parser.add_argument("im_or_folder", help="image or folder of images", default=None)

    parser.add_argument(
        "--background", action="store_true",
        help="The model will output predictions for the background class when set"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_detections_from_im(
    cfg,
    model,
    im,
    image_id,
    feat_blob_name,
    MIN_BOXES,
    MAX_BOXES,
    background=False,
    conf_thresh=0.2,
    bboxes=None,
):

    with c2_utils.NamedCudaScope(0):
        scores, cls_boxes, im_scale = infer_engine.im_detect_bbox(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=bboxes
        )
        box_features = workspace.FetchBlob(feat_blob_name)
        cls_prob = workspace.FetchBlob("gpu_0/cls_prob")
        rois = workspace.FetchBlob("gpu_0/rois")
        max_conf = np.zeros((rois.shape[0]))
        # unscale back to raw image space
        cls_boxes = rois[:, 1:5] / im_scale

        start_index = 1
        # Column 0 of the scores matrix is for the background class
        if background:
            start_index = 0
        for cls_ind in range(start_index, cls_prob.shape[1]):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(nms(dets, cfg.TEST.NMS))
            max_conf[keep] = np.where(
                cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep]
            )

        keep_boxes = np.where(max_conf >= conf_thresh)[0]
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
        # Predict the class label using the scores
        objects = np.argmax(cls_prob[keep_boxes][start_index:], axis=1)

    return box_features[keep_boxes]

    # return {
    #    "image_id": image_id,
    #    "image_h": np.size(im, 0),
    #    "image_w": np.size(im, 1),
    #    'num_boxes': len(keep_boxes),
    #    'boxes': base64.b64encode(cls_boxes[keep_boxes]),
    #    'features': base64.b64encode(box_features[keep_boxes]),
    #    'object': base64.b64encode(objects)
    # }


def extract_bboxes(bottom_up_csv_file):
    image_bboxes = {}

    with open(bottom_up_csv_file, "r") as tsv_in_file:
        reader = csv.DictReader(
            tsv_in_file, delimiter="\t", fieldnames=BOTTOM_UP_FIELDNAMES
        )
        for item in reader:
            item["num_boxes"] = int(item["num_boxes"])
            image_id = int(item["image_id"])
            image_w = float(item["image_w"])
            image_h = float(item["image_h"])

            bbox = np.frombuffer(
                base64.b64decode(item["boxes"]), dtype=np.float32
            ).reshape((item["num_boxes"], -1))

            image_bboxes[image_id] = bbox
    return image_bboxes


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    start = timeit.default_timer()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + "/*." + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    # extract bboxes from bottom-up attention model
    image_bboxes = {}
    if args.bbox_file is not None:
        image_bboxes = extract_bboxes(args.bbox_file)

    count = 0
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i, im_name in enumerate(im_list):
        im_base_name = os.path.basename(im_name)
        image_id = int(im_base_name.split(".")[0].split("_")[-1])  # for COCO
        if image_id % args.total_group == args.group_id:
            bbox = image_bboxes[image_id] if image_id in image_bboxes else None
            im = cv2.imread(im_name)
            if im is not None:
                outfile = os.path.join(
                    args.output_dir, im_base_name.replace("jpg", "npy")
                )
                lock_folder = outfile.replace("npy", "lock")
                if not os.path.exists(lock_folder) and os.path.exists(outfile):
                    continue
                if not os.path.exists(lock_folder):
                    os.makedirs(lock_folder)

                result = get_detections_from_im(
                    cfg,
                    model,
                    im,
                    image_id,
                    args.feat_name,
                    args.min_bboxes,
                    args.max_bboxes,
                    background=args.background,
                    bboxes=bbox,
                )

                np.save(outfile, result)
                os.rmdir(lock_folder)

            count += 1

            if count % 100 == 0:
                end = timeit.default_timer()
                epoch_time = end - start
                print("process {:d} images after {:.1f} s".format(count, epoch_time))


if __name__ == "__main__":
    workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    if args.group_id >= args.total_group:
        exit(
            "sharding group %d is greater than the total group %d"
            % (args.group_id, args.total_group)
        )

    main(args)
