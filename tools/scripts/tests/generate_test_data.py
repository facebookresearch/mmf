# Copyright (c) Facebook, Inc. and its affiliates.

"""
File can be used for generating test data

Takes in the DB file, features folder, image folder and will generate a test data
folder for a certain amount of samples in the following folder

output_folder/
    images/
        a.jpg
        b.jpg
        ...
    features/
        features.lmdb/
            data.mdb
            lock.mdb
        raw/
            a.npy
            a_info.npy
            b.npy
            b_info.npy
            ...
    db/
        train.jsonl
        dev.jsonl
        test.jsonl
"""
# Copyright (c) 2017-present, Facebook, Inc.

import argparse
import json
import os
import shutil

import numpy as np
from tools.scripts.features.lmdb_conversion import LMDBConversion


class TestDataBuilder:
    def __init__(self):
        parser = self.get_parser()
        self.args = parser.parse_args()

    def get_parser(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument(
            "--train_db_file",
            required=True,
            type=str,
            help="DB file that will be used for generating the test data for training.",
        )
        parser.add_argument(
            "--dev_db_file",
            required=True,
            type=str,
            help="DB file that will be used for generating the test data for "
            + "validation.",
        )
        parser.add_argument(
            "--num_samples",
            type=int,
            default=100,
            help="Number of samples to be extracted from the db file.",
        )
        parser.add_argument(
            "--train_images_folder",
            type=str,
            default=None,
            help="Images folder for training set",
        )
        parser.add_argument(
            "--dev_images_folder",
            type=str,
            default=None,
            help="Images folder for dev set",
        )
        parser.add_argument(
            "--train_features_folder",
            required=True,
            type=str,
            help="Features folder.",
            default=None,
        )
        parser.add_argument(
            "--dev_features_folder",
            required=True,
            type=str,
            help="Features folder.",
            default=None,
        )
        parser.add_argument(
            "--output_folder", required=True, type=str, help="Output folder."
        )
        return parser

    def build(self):
        self.generate_and_save_data(
            self.args.train_db_file,
            self.args.train_images_folder,
            self.args.train_features_folder,
            "train",
            self.args.num_samples,
            self.args.output_folder,
        )
        self.generate_and_save_data(
            self.args.dev_db_file,
            self.args.dev_images_folder,
            self.args.dev_features_folder,
            "dev",
            # Number of dev and test samples, we generate 1/10th of the training data
            self.args.num_samples // 10,
            self.args.output_folder,
        )
        # Test data is generated from dev data
        self.generate_and_save_data(
            self.args.dev_db_file,
            self.args.dev_images_folder,
            self.args.dev_features_folder,
            "test",
            self.args.num_samples // 10,
            self.args.output_folder,
        )

    def generate_and_save_data(
        self, db_file, image_folder, features_folder, name, num_samples, output_folder
    ):
        """This will generate features, db and images folder in proper format
        and save them to output folder

        Args:
            db_file (str): Path to DB file from which samples will be generated
            image_folder (str): Folder where images are present
            features_folder (str): Folder where raw features are present
            name (str): Type of the dataset set
            num_samples (int): Number of objects to be sampled
            output_folder (str): Path where output files will be stored
        """
        data, _ = self._load_db_file(db_file, num_samples)
        assert len(data) == num_samples and num_samples > 0
        image_paths = self._get_image_paths(data, image_folder)
        feature_paths = self._get_feature_paths(data, features_folder)

        image_output_folder = os.path.join(output_folder, "images/")
        os.makedirs(image_output_folder, exist_ok=True)

        for path in image_paths:
            shutil.copy(path, image_output_folder)

        features_output_folder = os.path.join(output_folder, "features", "raw/")
        os.makedirs(features_output_folder, exist_ok=True)

        for path in feature_paths:
            shutil.copy(path, features_output_folder)

        db_output_folder = os.path.join(output_folder, "db/")
        os.makedirs(db_output_folder, exist_ok=True)

        output = []
        for d in data:
            output.append(json.dumps(d))
        output = "\n".join(output)

        with open(os.path.join(db_output_folder, f"{name}.jsonl"), "w") as f:
            f.write(output)

        lmdb_folder = os.path.join(output_folder, "features")
        LMDBConversion.get_parser = mock_lmdb_parser(
            features_output_folder, lmdb_folder
        )
        lmdb_converter = LMDBConversion()
        lmdb_converter.execute()

    def _load_db_file(self, db_file: str, num_samples: int):
        """Load db file based on the format and return back a randomly sampled
        list of 'num_samples' objects from DB.

        Args:
            db_file (str): Path to DB file
            num_samples (int): Number of samples that will be generated

        Raises:
            ValueError: Raised if DB file is not among ".json|.jsonl|.npy"

        Returns:
            Tupe(List[Object], str): A tuple containing both the selected data and
            actual file path
        """
        file_type = None
        if db_file.endswith(".npy"):
            file_type = "npy"
            data = np.load(db_file, allow_pickle=True)
            selected_data = np.random.choice(data[1:], size=num_samples, replace=False)
        elif db_file.endswith(".jsonl"):
            file_type = "jsonl"
            with open(db_file) as f:
                data = []
                for item in f.readlines():
                    data.append(json.loads(item.strip("\n")))
                selected_data = np.random.choice(data, size=num_samples, replace=False)

        # Expecting JSON to be in COCOJSONFormat or contain "data" attribute
        elif db_file.endswith(".json"):
            file_type = "json"
            with open(db_file) as f:
                data = json.load(f)
                selected_data = np.random.choice(data, size=num_samples, replace=False)
        else:
            raise ValueError("Unexpected DB file type. Valid options {json|jsonl|npy}")

        return selected_data, file_type

    def _get_image_paths(self, data, image_folder):
        if image_folder is None:
            return []

        images = set()
        for item in data:
            possible_images = self._get_attrs(item)
            for image in possible_images:
                images.add(os.path.join(image_folder, image))

        return images

    def _get_feature_paths(self, data, feature_folder):
        if feature_folder is None:
            return []

        features = set()
        for item in data:
            possible_images = self._get_attrs(item)
            for image in possible_images:
                image = ".".join(image.split(".")[:-1])
                feature = image + ".npy"
                info = image + "_info.npy"
                features.add(os.path.join(feature_folder, feature))
                features.add(os.path.join(feature_folder, info))

        return features

    def _get_attrs(self, item):
        """Returns possible attribute that can point to image id

        Args:
            item (Object): Object from the DB

        Returns:
            List[str]: List of possible images that will be copied later
        """
        image = None
        pick = None
        attrs = self._get_possible_attrs()

        for attr in attrs:
            image = item.get(attr, None)
            if image is not None:
                pick = attr
                break

        if pick == "identifier":
            return [image + "-img0.jpg", image + "-img1.jpg"]
        elif pick == "image_name" or pick == "image_id":
            return [image + ".jpg"]
        else:
            return [image]

    def _get_possible_attrs(self):
        return [
            "Flickr30kID",
            "Flikr30kID",
            "identifier",
            "image_path",
            "image_name",
            "img",
            "image_id",
        ]


def mock_lmdb_parser(features_folder, output_folder):
    args = argparse.Namespace()
    args.mode = "convert"
    args.features_folder = features_folder
    args.lmdb_path = os.path.join(output_folder, "features.lmdb")

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.parse_args = lambda: args
    return lambda _: parser


if __name__ == "__main__":
    builder = TestDataBuilder()
    builder.build()
