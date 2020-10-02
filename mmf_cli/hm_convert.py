# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import hashlib
import os
import subprocess
import warnings
import zipfile

from mmf.utils.configuration import Configuration
from mmf.utils.download import copy, decompress, move
from mmf.utils.file_io import PathManager


class HMConverter:
    IMAGE_FILES = ["img.tar.gz", "img"]
    JSONL_PHASE_ONE_FILES = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    JSONL_PHASE_TWO_FILES = [
        "train.jsonl",
        "dev_seen.jsonl",
        "test_seen.jsonl",
        "dev_unseen.jsonl",
        "test_unseen.jsonl",
    ]
    POSSIBLE_CHECKSUMS = [
        "d8f1073f5fbf1b08a541cc2325fc8645619ab8ed768091fb1317d5c3a6653a77",
        "a424c003b7d4ea3f3b089168b5f5ea73b90a3ff043df4b8ff4d7ed87c51cb572",
        "6e609b8c230faff02426cf462f0c9528957b7884d68c60ebc26ff83846e5f80f",
        "c1363aae9649c79ae4abfdb151b56d3d170187db77757f3daa80856558ac367c",
    ]

    def __init__(self):
        self.parser = self.get_parser()
        self.args = self.parser.parse_args()
        self.configuration = Configuration()

    def assert_files(self, folder):
        files_needed = self.JSONL_PHASE_ONE_FILES
        phase_one = True
        for file in files_needed:
            try:
                assert PathManager.exists(
                    os.path.join(folder, "data", file)
                ), f"{file} doesn't exist in {folder}"
            except AssertionError:
                phase_one = False

        if not phase_one:
            files_needed = self.JSONL_PHASE_TWO_FILES
            for file in files_needed:
                assert PathManager.exists(
                    os.path.join(folder, "data", file)
                ), f"{file} doesn't exist in {folder}"
        else:
            warnings.warn(
                "You are on Phase 1 of the Hateful Memes Challenge. "
                "Please update to Phase 2"
            )

        files_needed = self.IMAGE_FILES

        exists = False

        for file in files_needed:
            exists = exists or PathManager.exists(os.path.join(folder, "data", file))

        if not exists:
            raise AssertionError("Neither img or img.tar.gz exists in current zip")

        return phase_one

    def get_parser(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument(
            "--zip_file",
            required=True,
            type=str,
            help="Zip file downloaded from the DrivenData",
        )

        parser.add_argument(
            "--password", required=True, type=str, help="Password for the zip file"
        )
        parser.add_argument(
            "--move", required=None, type=int, help="Move data dir to mmf cache dir"
        )
        parser.add_argument(
            "--mmf_data_folder", required=None, type=str, help="MMF Data folder"
        )
        parser.add_argument(
            "--bypass_checksum",
            required=None,
            type=int,
            help="Pass 1 if you want to skip checksum",
        )
        return parser

    def convert(self):
        config = self.configuration.get_config()
        data_dir = config.env.data_dir

        if self.args.mmf_data_folder:
            data_dir = self.args.mmf_data_folder

        bypass_checksum = False
        if self.args.bypass_checksum:
            bypass_checksum = bool(self.args.bypass_checksum)

        print(f"Data folder is {data_dir}")
        print(f"Zip path is {self.args.zip_file}")

        base_path = os.path.join(data_dir, "datasets", "hateful_memes", "defaults")

        images_path = os.path.join(base_path, "images")
        PathManager.mkdirs(images_path)

        move_dir = False
        if self.args.move:
            move_dir = bool(self.args.move)

        if not bypass_checksum:
            self.checksum(self.args.zip_file, self.POSSIBLE_CHECKSUMS)

        src = self.args.zip_file
        dest = images_path
        if move_dir:
            print(f"Moving {src}")
            move(src, dest)
        else:
            print(f"Copying {src}")
            copy(src, dest)

        print(f"Unzipping {src}")
        self.decompress_zip(
            dest, fname=os.path.basename(src), password=self.args.password
        )

        phase_one = self.assert_files(images_path)

        annotations_path = os.path.join(base_path, "annotations")
        PathManager.mkdirs(annotations_path)
        annotations = (
            self.JSONL_PHASE_ONE_FILES
            if phase_one is True
            else self.JSONL_PHASE_TWO_FILES
        )

        for annotation in annotations:
            print(f"Moving {annotation}")
            src = os.path.join(images_path, "data", annotation)
            dest = os.path.join(annotations_path, annotation)
            move(src, dest)

        images = self.IMAGE_FILES

        for image_file in images:
            src = os.path.join(images_path, "data", image_file)
            if PathManager.exists(src):
                print(f"Moving {image_file}")
            else:
                continue
            dest = os.path.join(images_path, image_file)
            move(src, dest)
            if src.endswith(".tar.gz"):
                decompress(dest, fname=image_file, delete_original=False)

    def checksum(self, file, hashes):
        sha256_hash = hashlib.sha256()
        destination = file

        with PathManager.open(destination, "rb") as f:
            print("Starting checksum for {}".format(os.path.basename(file)))
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
            if sha256_hash.hexdigest() not in hashes:
                # remove_dir(download_path)
                raise AssertionError(
                    f"Checksum of downloaded file does not match the expected "
                    + "checksum. Please try again."
                )
            else:
                print("Checksum successful")

    def decompress_zip(self, dest, fname, password=None):
        path = os.path.join(dest, fname)
        print("Extracting the zip can take time. Sit back and relax.")
        try:
            # Python's zip file module is very slow with password encrypted files
            # Try command line
            command = ["unzip", "-o", "-q", "-d", dest]
            if password:
                command += ["-P", password]
            command += [path]
            subprocess.run(command, check=True)
        except Exception:
            obj = zipfile.ZipFile(path, "r")
            if password:
                obj.setpassword(password.encode("utf-8"))
            obj.extractall(path=dest)
            obj.close()


def main():
    converter = HMConverter()
    converter.convert()


if __name__ == "__main__":
    main()
