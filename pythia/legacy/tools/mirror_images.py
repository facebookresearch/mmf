# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# All paths need to be updated

import json
import os
from multiprocessing.dummy import Pool as ThreadPool

from PIL import Image, ImageOps

split = "val2014"
image_paths = []


def mirror_image(image_path):
    img = Image.open(image_path)
    mirror_img = ImageOps.mirror(img)
    image_name = image_path.split("/")[-1]
    fh = "data/" + split
    fh = os.path.join(fh, image_name)
    mirror_img.save(fh, "JPEG")


with open("./COCO/060817/annotations/instances_val2014.json") as f:
    data = json.load(f)
    for item in data["images"]:
        image_id = int(item["id"])
        filepath = os.path.join("val2014/", item["file_name"])
        image_paths.append(filepath)

pool = ThreadPool(10)
results = pool.map(mirror_image, image_paths)
