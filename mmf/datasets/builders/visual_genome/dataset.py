# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import json

import torch
from mmf.common.sample import Sample, SampleList
from mmf.datasets.builders.vqa2 import VQA2Dataset
from mmf.datasets.databases.scene_graph_database import SceneGraphDatabase


_CONSTANTS = {"image_id_key": "image_id"}


class VisualGenomeDataset(VQA2Dataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            config,
            dataset_type,
            imdb_file_index,
            dataset_name="visual_genome",
            *args,
            **kwargs
        )

        self._return_scene_graph = config.return_scene_graph
        self._return_objects = config.return_objects
        self._return_relationships = config.return_relationships
        self._no_unk = config.get("no_unk", False)
        self.scene_graph_db = None

        build_scene_graph_db = (
            self._return_scene_graph
            or self._return_objects
            or self._return_relationships
        )

        if build_scene_graph_db:
            scene_graph_file = config.scene_graph_files[dataset_type][imdb_file_index]
            scene_graph_file = self._get_absolute_path(scene_graph_file)
            self.scene_graph_db = SceneGraphDatabase(config, scene_graph_file)

    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info = self._preprocess_answer(sample_info)
        sample_info["question_id"] = sample_info["id"]
        if self._check_unk(sample_info):
            return self.load_item((idx + 1) % len(self.annotation_db))

        current_sample = super().load_item(idx)
        current_sample = self._load_scene_graph(idx, current_sample)

        return current_sample

    def _get_image_id(self, idx):
        return self.annotation_db[idx][_CONSTANTS["image_id_key"]]

    def _get_image_info(self, idx):
        # Deep copy so that we can directly update the nested dicts
        return copy.deepcopy(self.scene_graph_db[self._get_image_id(idx)])

    def _preprocess_answer(self, sample_info):
        sample_info["answers"] = [
            self.vg_answer_preprocessor(
                {"text": sample_info["answers"][0]},
                remove=["?", ",", ".", "a", "an", "the"],
            )["text"]
        ]

        return sample_info

    def _check_unk(self, sample_info):
        if not self._no_unk:
            return False
        else:
            index = self.answer_processor.word2idx(sample_info["answers"][0])
            return index == self.answer_processor.answer_vocab.UNK_INDEX

    def _load_scene_graph(self, idx, sample):
        if self.scene_graph_db is None:
            return sample

        image_info = self._get_image_info(idx)
        regions = image_info["regions"]

        objects, object_map = self._load_objects(idx)

        if self._return_objects:
            sample.objects = objects

        relationships, relationship_map = self._load_relationships(idx, object_map)

        if self._return_relationships:
            sample.relationships = relationships

        regions, _ = self._load_regions(idx, object_map, relationship_map)

        if self._return_scene_graph:
            sample.scene_graph = regions

        return sample

    def _load_objects(self, idx):
        image_info = self._get_image_info(idx)
        image_height = image_info["height"]
        image_width = image_info["width"]
        object_map = {}
        objects = []

        for obj in image_info["objects"]:
            obj["synsets"] = self.synset_processor({"tokens": obj["synsets"]})["text"]
            obj["names"] = self.name_processor({"tokens": obj["names"]})["text"]
            obj["height"] = obj["h"] / image_height
            obj.pop("h")
            obj["width"] = obj["w"] / image_width
            obj.pop("w")
            obj["y"] /= image_height
            obj["x"] /= image_width
            obj["attributes"] = self.attribute_processor({"tokens": obj["attributes"]})[
                "text"
            ]
            obj = Sample(obj)
            object_map[obj["object_id"]] = obj
            objects.append(obj)
        objects = SampleList(objects)

        return objects, object_map

    def _load_relationships(self, idx, object_map):
        if self._return_relationships is None and self._return_scene_graph is None:
            return None, None

        image_info = self._get_image_info(idx)
        relationship_map = {}
        relationships = []

        for relationship in image_info["relationships"]:
            relationship["synsets"] = self.synset_processor(
                {"tokens": relationship["synsets"]}
            )["text"]
            relationship["predicate"] = self.predicate_processor(
                {"tokens": relationship["predicate"]}
            )["text"]
            relationship["object"] = object_map[relationship["object_id"]]
            relationship["subject"] = object_map[relationship["subject_id"]]

            relationship = Sample(relationship)
            relationship_map[relationship["relationship_id"]] = relationship
            relationships.append(relationship)

        relationships = SampleList(relationships)
        return relationships, relationship_map

    def _load_regions(self, idx, object_map, relationship_map):
        if self._return_scene_graph is None:
            return None, None

        image_info = self._get_image_info(idx)
        image_height = image_info["height"]
        image_width = image_info["width"]
        region_map = {}
        regions = []

        for region in image_info["regions"]:
            for synset in region["synsets"]:
                synset["entity_name"] = self.name_processor(
                    {"tokens": [synset["entity_name"]]}
                )["text"]
                synset["synset_name"] = self.synset_processor(
                    {"tokens": [synset["synset_name"]]}
                )["text"]

            region["height"] /= image_height
            region["width"] /= image_width
            region["y"] /= image_height
            region["x"] /= image_width

            relationships = []
            objects = []

            for relationship_idx in region["relationships"]:
                relationships.append(relationship_map[relationship_idx])

            for object_idx in region["objects"]:
                objects.append(object_map[object_idx])

            region["relationships"] = relationships
            region["objects"] = objects
            region["phrase"] = self.text_processor({"text": region["phrase"]})["text"]

            region = Sample(region)
            region_map[region["region_id"]] = region
            regions.append(region)

        regions = SampleList(regions)
        return regions, region_map
