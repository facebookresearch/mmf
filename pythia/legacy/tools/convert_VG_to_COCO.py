# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import json
import os

import _pickle

COMMON_ATTRIBUTES = set(
    [
        "white",
        "black",
        "blue",
        "green",
        "red",
        "brown",
        "yellow",
        "small",
        "large",
        "silver",
        "wooden",
        "orange",
        "gray",
        "grey",
        "metal",
        "pink",
        "tall",
        "long",
        "dark",
    ]
)


def extract_category_map(objects_list_file):
    with open(objects_list_file, "r") as f:
        lines = f.readlines()
    obj_list = [line.strip() for line in lines]
    id_map = {}

    for i, line in enumerate(obj_list):
        objects = line.split(",")
        for obj in objects:
            id_map[obj] = i
    return id_map


def clean_category(id_map):
    clean_id_map = {}
    for obj_name, obj_id in id_map.items():
        if obj_id not in clean_id_map:
            clean_id_map[obj_id] = obj_name
    return clean_id_map


def get_segmantation(bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    return [x, y, x, y + h, x + w, y + h, x + w, y]


def get_area(bbox):
    return bbox[2] * bbox[3]


def clean_string(string):
    string = string.lower().strip()
    if len(string) >= 1 and string[-1] == ".":
        return string[:-1].strip()
    return string


def clean_objects(string, common_attributes):
    """ Return object and attribute lists """
    string = clean_string(string)
    words = string.split()
    if len(words) > 1:
        prefix_words_are_adj = True
        for att in words[:-1]:
            if att not in common_attributes:
                prefix_words_are_adj = False
        if prefix_words_are_adj:
            return words[-1:], words[:-1]
        else:
            return [string], []
    else:
        return [string], []


def clean_attributes(string):
    """ Return attribute list """
    string = clean_string(string)
    if string == "black and white":
        return [string]
    else:
        return [word.lower().strip() for word in string.split(" and ")]


class COCO_annotation:
    def __init__(self, obj_cat_id_map, att_cat_id_map):
        self.annotations = []
        self.number_of_object = 0

        self.obj_cat_id_map = obj_cat_id_map
        self.att_cat_id_map = att_cat_id_map
        print(
            "total number of objects is %d, total number of attributes is %d"
            % (len(self.obj_cat_id_map), len(self.att_cat_id_map))
        )

    def add_annotation(self, obj_id, image_id, bbox, att_ids):
        segmentation = [get_segmantation(bbox)]
        area = get_area(bbox)
        self.number_of_object += 1
        annotation = {
            "id": self.number_of_object,
            "category_id": obj_id,
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": bbox,
            "ignore": 0,
        }
        if len(att_ids) > 0:
            annotation["attribute_ids"] = att_ids
        self.annotations.append(annotation)

    def summary(self):
        obj_clean_map = clean_category(self.obj_cat_id_map)
        self.categories = [
            {"supercategory": "obj", "id": cat_id, "name": cat_name}
            for cat_id, cat_name in obj_clean_map.items()
        ]
        att_clean_map = clean_category(self.att_cat_id_map)
        self.att_categories = [
            {"supercategory": "att", "id": cat_id, "name": cat_name}
            for cat_id, cat_name in att_clean_map.items()
        ]

    def add_images(self, images, image_group):
        self.images = []
        for image in images:
            image_name = os.path.basename(image["url"])
            if image_name not in image_group:
                continue

            new_image = {
                "file_name": image_name,
                "id": image["image_id"],
                "height": image["height"],
                "width": image["width"],
            }
            self.images.append(new_image)

    def print(self, writer):
        tmp = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
            "attCategories": self.att_categories,
        }
        return json.dump(tmp, writer)


def convert_2_object_and_att(
    VG_attributes, image_group, category_id_map, att_cat_id_maps
):
    cocoAnnotation = COCO_annotation(category_id_map, att_cat_id_maps)

    count = 0
    for image in VG_attributes:

        count += 1

        if count % 10000 == 0:
            print("process %d images" % count)

        image_id = image["image_id"]
        image_name = str(image_id) + ".jpg"
        if image_name not in image_group:
            continue

        for att in image["attributes"]:
            atts = [] if "attributes" not in att else att["attributes"]
            bbox = [att["x"], att["y"], att["w"], att["h"]]
            object_names = att["names"]

            obj = object_names[0]
            objs, atts_from_name = clean_objects(obj, COMMON_ATTRIBUTES)
            obj_ids = []
            for obj_tmp in objs:
                if obj_tmp in cocoAnnotation.obj_cat_id_map:
                    obj_ids.append(cocoAnnotation.obj_cat_id_map[obj_tmp])
            if len(obj_ids) == 0:
                continue

            obj_id = obj_ids[0]

            atts += atts_from_name
            att_ids = []
            for att in atts:
                clean_atts = clean_attributes(att)
                for clean_att in clean_atts:
                    if clean_att in cocoAnnotation.att_cat_id_map:
                        clean_att_id = cocoAnnotation.att_cat_id_map[clean_att]
                        att_ids.append(clean_att_id)

            cocoAnnotation.add_annotation(obj_id, image_id, bbox, att_ids)

    return cocoAnnotation


if __name__ == "__main__":
    VG_attributes_file = "attributes.json"
    VG_image_file = "image_data.json"
    VG_object_and_atrributes_in_COCO_file = "object_and_attributes_%s.json"

    image_id_file = "%s.pkl"
    attributes_list_file = "attributes_vocab.txt"
    objects_list_file = "objects_vocab.txt"

    with open(VG_attributes_file, "r") as f:
        VG_attributes = json.load(f)

    obj_cat_id_maps = extract_category_map(objects_list_file)
    att_cat_id_maps = extract_category_map(attributes_list_file)

    for label in ["train", "val", "test"]:

        print("process " + label + "...")
        with open(image_id_file % label, "rb") as f:
            image_ids = _pickle.load(f)

        VG_object_and_atrributes_in_COCO = convert_2_object_and_att(
            VG_attributes, image_ids, obj_cat_id_maps, att_cat_id_maps
        )

        # extract image meta data
        with open(VG_image_file, "r") as f:
            VG_image = json.load(f)

        VG_object_and_atrributes_in_COCO.summary()
        VG_object_and_atrributes_in_COCO.add_images(VG_image, image_ids)

        with open(VG_object_and_atrributes_in_COCO_file % label, "w") as w:
            VG_object_and_atrributes_in_COCO.print(w)
