# Copyright (c) Facebook, Inc. and its affiliates.
import json
import torch
import copy

from pythia.tasks.vqa.vqa2 import VQA2Dataset
from pythia.tasks.scene_graph_database import SceneGraphDatabase
from pythia.common.sample import Sample, SampleList

_CONSTANTS = {
    "image_id_key": "image_id"
}


class VisualGenomeDataset(VQA2Dataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(dataset_type, imdb_file_index, config, *args, **kwargs)
        self._name = "visual_genome"

        self._return_scene_graph = config.return_scene_graph
        self._return_objects = config.return_objects
        self._return_relationships = config.return_relationships
        self.scene_graph_db = None

        build_scene_graph_db = (
            self._return_scene_graph or self._return_objects
            or self._return_relationships
        )

        if build_scene_graph_db:
            scene_graph_file = config.scene_graph_files[dataset_type][imdb_file_index]
            scene_graph_file = self._get_absolute_path(scene_graph_file)
            self.scene_graph_db = SceneGraphDatabase(scene_graph_file)

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        sample_info = self._preprocess_answer(sample_info)
        current_sample = Sample()

        text_processor_argument = {"text": sample_info["question"]}

        processed_question = self.text_processor(text_processor_argument)

        current_sample.text = processed_question["text"]
        current_sample.question_id = torch.tensor(
            sample_info["id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["image_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["image_id"]

        current_sample.text_len = torch.tensor(
            len(processed_question["text"]), dtype=torch.int
        )

        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)

        # Add details for OCR like OCR bbox, vectors, tokens here
        current_sample = self.add_ocr_details(sample_info, current_sample)
        # Depending on whether we are using soft copy this can add
        # dynamic answer space
        current_sample = self.add_answer_info(sample_info, current_sample)
        current_sample = self._load_scene_graph(idx, current_sample)

        return current_sample

    def _get_image_id(self, idx):
        return self.imdb[idx][_CONSTANTS["image_id_key"]]

    def _get_image_info(self, idx):
        # Deep copy so that we can directly update the nested dicts
        return copy.deepcopy(self.scene_graph_db[self._get_image_id(idx)])

    def _preprocess_answer(self, sample_info):
        sample_info["answers"] = [self.vg_answer_preprocessor({
            "text": sample_info["answers"][0]
        }, remove=["?", ",", "."])["text"]]

        return sample_info

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
            obj["attributes"] = self.attribute_processor({
                "tokens": obj["attributes"]
            })["text"]
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
            relationship["synsets"] = self.synset_processor({
                "tokens": relationship["synsets"]
            })["text"]
            relationship["predicate"] = self.predicate_processor({
                "tokens": relationship["predicate"]
            })["text"]
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
                synset["entity_name"] = self.name_processor({
                    "tokens": [synset["entity_name"]]
                })["text"]
                synset["synset_name"] = self.synset_processor({
                    "tokens": [synset["synset_name"]]
                })["text"]

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
            region["phrase"] = self.text_processor({
                "text": region["phrase"]
            })["text"]

            region = Sample(region)
            region_map[region["region_id"]] = region
            regions.append(region)

        regions = SampleList(regions)
        return regions, region_map
