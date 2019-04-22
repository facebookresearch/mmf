# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import json
import pickle

if __name__ == "__main__":
    val_annotation_file = "v2_mscoco_val2014_annotations.json"
    minival_id_file = "data/vqa_v2.0/minival_ids.pkl"
    minival_annotation_file = "v2_mscoco_minival2014_annotations.json"

    with open(minival_id_file, "rb") as f:
        q_im_ids = pickle.load(f)

    minival_ids = [x[1] for x in q_im_ids]

    with open(val_annotation_file, "r") as f:
        file_info = json.load(f)
        annotations = file_info["annotations"]
        info = file_info["info"]
        data_subtype = file_info["data_subtype"]
        license_info = file_info["license"]

    minival_annotations = [a for a in annotations if a["question_id"] in minival_ids]

    minival_info = {
        "data_subtype": data_subtype,
        "license": license_info,
        "info": info,
        "annotations": minival_annotations,
    }

    with open(minival_annotation_file, "w") as w:
        json.dump(minival_info, w)
