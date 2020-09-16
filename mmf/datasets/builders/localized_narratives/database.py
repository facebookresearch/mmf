# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import json
from typing import Dict, Generator, List, NamedTuple

from mmf.datasets.databases.annotation_database import AnnotationDatabase
from mmf.utils.general import get_absolute_path


class TimedPoint(NamedTuple):
    x: float
    y: float
    t: float


class TimedUtterance(NamedTuple):
    utterance: str
    start_time: float
    end_time: float


class LocalizedNarrative(NamedTuple):
    dataset_id: str
    image_id: str
    annotator_id: int
    caption: str
    timed_caption: List[TimedUtterance]
    traces: List[List[TimedPoint]]
    voice_recording: str

    def __repr__(self):
        truncated_caption = (
            self.caption[:60] + "..." if len(self.caption) > 63 else self.caption
        )
        truncated_timed_caption = self.timed_caption[0].__str__()
        truncated_traces = self.traces[0][0].__str__()
        return (
            f"{{\n"
            f" dataset_id: {self.dataset_id},\n"
            f" image_id: {self.image_id},\n"
            f" annotator_id: {self.annotator_id},\n"
            f" caption: {truncated_caption},\n"
            f" timed_caption: [{truncated_timed_caption}, ...],\n"
            f" traces: [[{truncated_traces}, ...], ...],\n"
            f" voice_recording: {self.voice_recording}\n"
            f"}}"
        )


class LocalizedNarrativesAnnotationDatabase(AnnotationDatabase):
    def __init__(self, config, path, *args, **kwargs):
        super().__init__(config, path, *args, **kwargs)

    def load_annotation_db(self, path):
        data = []
        with open(path) as f:
            for line in f:
                annotation = json.loads(line)
                loc_narr = LocalizedNarrative(**annotation)
                data.append(
                    {
                        **annotation,
                        "feature_path": self._feature_path(
                            loc_narr.dataset_id, loc_narr.image_id
                        ),
                    }
                )
        self.data = data

    def _feature_path(self, dataset_id, image_id):
        # TODO: @sash update with coco/openimages
        if dataset_id == "Flick30k" or dataset_id == "ADE20k":
            return image_id + ".npy"
