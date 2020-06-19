# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import json
import os

from mmf.utils.text import tokenize


class IMDBBuilder:
    def __init__(self):
        self.args = self.get_args()

    def get_args(self):
        parser = argparse.ArgumentParser("Build IMDB for VisDial")
        parser.add_argument(
            "-o",
            "--out_file",
            type=str,
            default="./imdb.npy",
            help="Output file for IMDB",
        )
        parser.add_argument(
            "-i",
            "--image_root",
            type=str,
            default="./COCO",
            help="Image directory for COCO",
        )
        parser.add_argument(
            "-v", "--version", type=float, default=0.9, help="Visdial version"
        )
        parser.add_argument(
            "-d",
            "--data_dir",
            type=str,
            default="./visdial",
            help="Directory which contains visdial jsons",
        )
        parser.add_argument(
            "-s",
            "--set_type",
            type=str,
            default="train",
            help="Dataset type train|val|test",
        )

        return parser.parse_args()

    def get_id_to_path_dict(self):
        id2path = {}
        globs = glob.iglob(os.path.join(self.args.image_root, "*", "*.npy"))
        # NOTE: based on assumption that image_id is unique across all splits
        for image_path in globs:
            path = "/".join(image_path.split("/")[-2:])
            image_id = int(image_path[-16:-4])
            id2path[image_id] = path

        return id2path

    def build(self):
        visdial_json_file = os.path.join(
            self.args.data_dir,
            "visdial_%.1f_%s.json" % (self.args.version, self.args.set_type),
        )
        data = None

        with open(visdial_json_file, "r") as f:
            data = json.load(f)["data"]

        final_questions = self.get_tokens(data["questions"])
        final_answers = self.get_tokens(data["answers"])
        dialogs = data["dialogs"]

        dialogs_with_features = self.parse_dialogs(dialogs)

        imdb = {
            "questions": final_questions,
            "answers": final_answers,
            "dialogs": dialogs_with_features,
        }

        self.save_imdb(imdb)

    def save_imdb(self, imdb):
        with open(self.args.out_file, "w") as f:
            json.dump(imdb, f)

    def get_tokens(self, sentences):
        if not isinstance(sentences, list):
            sentences = [sentences]
        final_sentences = []
        for _, sentence in enumerate(sentences):
            tokens = tokenize(sentence)
            final_sentences.append(tokens)

        return final_sentences

    def parse_dialogs(self, dialogs):
        id2path = self.get_id_to_path_dict()

        for dialog in dialogs:
            image_id = dialog["image_id"]
            image_feature_path = id2path[image_id]
            dialog["image_feature_path"] = image_feature_path
            dialog["caption"] = self.get_tokens(dialog["caption"])

        return dialogs


if __name__ == "__main__":
    imdb_builder = IMDBBuilder()
    imdb_builder.build()
