# Copyright (c) 2017-present, Facebook, Inc.

import argparse
import glob
import os
import pickle

import lmdb
import numpy as np
import tqdm


class LMDBConversion:
    def __init__(self):
        self.args = self.get_parser().parse_args()

    def get_parser(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument(
            "--mode",
            required=True,
            type=str,
            help="Mode can either be `convert` (for conversion of \n"
            + "features to an LMDB file) or `extract` (extract \n"
            + "raw features from a LMDB file)",
        )
        parser.add_argument(
            "--lmdb_path", required=True, type=str, help="LMDB file path"
        )
        parser.add_argument(
            "--features_folder", required=True, type=str, help="Features folder"
        )
        return parser

    def convert(self):
        env = lmdb.open(self.args.lmdb_path, map_size=1099511627776)
        id_list = []
        features = glob.glob(os.path.join(self.args.features_folder, "*_info.npy"))
        with env.begin(write=True) as txn:
            for infile in tqdm.tqdm(features):
                reader = np.load(infile, allow_pickle=True)
                item = {}
                item["image_id"] = int(
                    os.path.basename(infile).split("_info")[0].split("_")[-1]
                )
                img_id = str(item["image_id"]).encode()
                id_list.append(img_id)
                item["image_height"] = reader.item().get("image_height")
                item["image_width"] = reader.item().get("image_width")
                item["num_boxes"] = reader.item().get("num_boxes")
                item["objects"] = reader.item().get("objects")
                item["cls_prob"] = reader.item().get("cls_prob")
                item["bbox"] = reader.item().get("bbox")
                item["features"] = np.load(
                    infile.split("_info")[0] + ".npy", allow_pickle=True
                )
                txn.put(img_id, pickle.dumps(item))
            txn.put("keys".encode(), pickle.dumps(id_list))

    def extract(self):
        os.makedirs(self.args.features_folder, exist_ok=True)
        env = lmdb.open(
            self.args.lmdb_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            _image_ids = pickle.loads(txn.get("keys".encode()))
            for img_id in tqdm.tqdm(_image_ids):
                item = pickle.loads(txn.get(img_id))
                img_id = img_id.decode("utf-8")
                tmp_dict = {}
                tmp_dict["image_id"] = img_id
                tmp_dict["bbox"] = item["bbox"]
                tmp_dict["num_boxes"] = item["num_boxes"]
                tmp_dict["image_height"] = item["image_width"]
                tmp_dict["image_width"] = item["image_width"]
                tmp_dict["objects"] = item["objects"]
                tmp_dict["cls_prob"] = item["cls_prob"]

                info_file_base_name = str(img_id) + "_info.npy"
                file_base_name = str(img_id) + ".npy"

                np.save(
                    os.path.join(self.args.features_folder, file_base_name),
                    item["features"],
                )
                np.save(
                    os.path.join(self.args.features_folder, info_file_base_name),
                    tmp_dict,
                )

    def execute(self):
        if self.args.mode == "convert":
            self.convert()
        elif self.args.mode == "extract":
            self.extract()
        else:
            raise ValueError("mode must be either `convert` or `extract` ")


if __name__ == "__main__":
    lmdb_converter = LMDBConversion()
    lmdb_converter.execute()
