# Copyright (c) Facebook, Inc. and its affiliates.
import json

from pythia.scripts.extract_vocabulary import ExtractVocabulary


class ExtractVisdialVocabulary(ExtractVocabulary):
    def __init__(self):
        super(ExtractVisdialVocabulary, self).__init__()

    def get_text(self):
        text = []

        for input_file in self.input_files:
            with open(input_file, "r") as f:
                f_json = json.load(f)
                # Add 'questions' from visdial
                text += f_json["data"]["questions"]
                # Add 'answers' from visdial
                text += f_json["data"]["answers"]

                for dialog in f_json["data"]["dialogs"]:
                    text += [dialog["caption"]]
        return text


if __name__ == "__main__":
    extractor = ExtractVisdialVocabulary()
    extractor.extract()
