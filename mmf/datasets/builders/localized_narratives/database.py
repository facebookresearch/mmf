# Copyright (c) Facebook, Inc. and its affiliates.
import json
from typing import List, NamedTuple

import torch
from mmf.datasets.databases.annotation_database import AnnotationDatabase


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
                data_detail = {
                    "image_id": loc_narr.image_id,
                    "feature_path": self._feature_path(
                        loc_narr.dataset_id, loc_narr.image_id
                    ),
                }
                if (
                    hasattr(self.config, "trace_mode")
                    and self.config.trace_mode is not None
                ):
                    data_detail["utterances"] = self._utterances(loc_narr.timed_caption)
                    data_detail["utterance_times"] = self._utterance_times(
                        loc_narr.timed_caption
                    )
                    data_detail["traces"] = self._traces(loc_narr.traces)
                else:
                    data_detail["caption"] = loc_narr.caption

                data.append(data_detail)

        self.data = data

    def _utterances(self, timed_caption):
        return [caption["utterance"] for caption in timed_caption]

    def _utterance_times(self, timed_caption):
        times = [
            [caption["start_time"], caption["end_time"]] for caption in timed_caption
        ]
        return torch.FloatTensor(times)

    def _traces(self, traces):
        all_traces = [t for trace in traces for t in trace]
        all_traces = [[t["x"], t["y"], t["t"]] for t in all_traces]
        return torch.FloatTensor(all_traces)

    def _feature_path(self, dataset_id, image_id):
        if "mscoco" in dataset_id.lower():
            return image_id.rjust(12, "0") + ".npy"

        return image_id + ".npy"
