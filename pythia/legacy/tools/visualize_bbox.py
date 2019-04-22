# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import base64
import csv
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


plt.switch_backend("agg")

matplotlib.use("Agg")

plt.switch_backend("agg")

csv.field_size_limit(sys.maxsize)

FIELDNAMES = [
    "image_id",
    "image_w",
    "image_h",
    "num_boxes",
    "boxes",
    "features",
    "object",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--image_name_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--csv_file", type=str, required=True)
    args = parser.parse_args()

    return args


def vis_detections(im, class_name, bboxes, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect="equal")
    for i in inds:
        bbox = dets[i, :4]
        ax.add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="red",
                linewidth=3.5,
            )
        )
        """ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')"""

    """ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)"""
    plt.axis("off")
    plt.tight_layout()
    plt.draw()


def plot_bboxes(im, im_file, bboxes, out_dir):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect="equal")
    nboexes, _ = bboxes.shape

    for i in range(nboexes):
        bbox = bboxes[i, :4]

        ax.add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="red",
                linewidth=2.5,
            )
        )
    plt.axis("off")
    plt.tight_layout()
    plt.draw()

    out_file = os.path.join(out_dir, im_file.replace(".jpg", "_demo.jpg"))
    plt.savefig(out_file)


if __name__ == "__main__":
    args = parse_args()

    csvFile = args.csv_file

    image_bboxes = {}
    with open(csvFile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES)
        for item in reader:
            item["num_boxes"] = int(item["num_boxes"])
            image_id = item["image_id"]
            image_w = float(item["image_w"])
            image_h = float(item["image_h"])

            bboxes = np.frombuffer(
                base64.b64decode(item["boxes"]), dtype=np.float32
            ).reshape((item["num_boxes"], -1))
            image_bboxes[image_id] = bboxes

    out_dir = args.out_dir
    img_dir = args.img_dir

    with open(args.image_name_file, "r") as f:
        content = f.readlines()

    imgs = [x.strip() for x in content]

    for image_name in imgs:
        image_path = os.path.join(img_dir, image_name + ".jpg")
        im = cv2.imread(image_path)
        image_id = str(int(image_name.split("_")[2]))
        bboxes = image_bboxes[image_id]
        plot_bboxes(im, image_name, bboxes, out_dir)
