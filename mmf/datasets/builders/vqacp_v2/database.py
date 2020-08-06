# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import json

from mmf.datasets.builders.okvqa.database import OKVQAAnnotationDatabase
from mmf.utils.file_io import PathManager


class VQACPv2AnnotationDatabase(OKVQAAnnotationDatabase):
    def __init__(self, config, path, *args, **kwargs):
        super().__init__(config, path, *args, **kwargs)

    def load_annotation_db(self, path):
        # Expect two paths, one to questions and one to annotations
        assert (
            len(path) == 2
        ), "VQACPv2 requires 2 paths; one to questions and one to annotations"

        with PathManager.open(path[0]) as f:
            path_0 = json.load(f)
        with PathManager.open(path[1]) as f:
            path_1 = json.load(f)

        if "annotations" in path[0]:
            annotations = path_0
            questions = path_1
        else:
            annotations = path_1
            questions = path_0

        # Convert to linear format
        data = []
        question_dict = {}
        for question in questions:
            question_dict[question["question_id"]] = question["question"]

        for annotation in annotations:
            annotation["question"] = question_dict[annotation["question_id"]]
            answers = []
            for answer in annotation["answers"]:
                answers.append(answer["answer"])
            annotation["answers"] = answers
            data.append(copy.deepcopy(annotation))

        self.data = data
