import collections
import os
import random

from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.databases.image_database import ImageDatabase

from mmf.datasets.builders.retrieval.flickr30k_dataset import Flickr30kAnnotationDatabase

ANNOTATIONS_DATASET = {
    'flickr30k': Flickr30kAnnotationDatabase
}


class RetrievalDataset(BaseDataset):

    def __init__(self, config, dataset_type, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "retrieval"
        super().__init__(name, config, dataset_type)

        self.annotation_class = config.get("annotations_parser", 'flickr30k')

        self.load()

    def init_processors(self):
        super().init_processors()
        # Assign transforms to the image_db
        self.image_db.transform = self.image_processor

    def load(self):
        # Annotations
        splits_path = self._get_path_by_attribute(
            self.config, "annotations"
        )
        ann_path = self._get_annotations_extra_path(self.config)
        annotation_class = ANNOTATIONS_DATASET[self.annotation_class]
        self.annotation_db = annotation_class(self.config, splits_path, ann_path)

        # Images
        images_path = self._get_path_by_attribute(
            self.config, "images"
        )
        self.image_db = ImageDatabase(self.config, images_path, annotation_db=self.annotation_db)

    def __len__(self):
        return len(self.annotation_db)

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        sentence = random.sample(sample_info["sentences"], 1)[0]
        processed_sentence = self.text_processor({"text": sentence})

        current_sample.text = processed_sentence["text"]
        if "input_ids" in processed_sentence:
            current_sample.update(processed_sentence)

        current_sample.image = self.image_db[idx]["images"][0]

        return current_sample

    def _get_path_by_attribute(self, config, attribute):
        if attribute not in config:
            raise ValueError(f"{attribute} not present in config")

        config = config.get(attribute, None)

        if (
            self.dataset_type not in config
            or len(config.get(self.dataset_type, [])) == 0
        ):
            raise ValueError(f"No {attribute} present for type {self.dataset_type}")

        path = config[self.dataset_type]

        if isinstance(path, str):
            selected_path = path
        else:
            assert isinstance(path, collections.abc.MutableSequence)
            selected_path = path[0]

        selected_path = self._add_root_dir(selected_path)

        return selected_path

    def _get_annotations_extra_path(self, config):

        config = config.get("annotations", None)

        path = config["extras"]

        if isinstance(path, str):
            selected_path = path
        else:
            assert isinstance(path, collections.abc.MutableSequence)
            selected_path = path[0]

        selected_path = self._add_root_dir(selected_path)

        return selected_path

    def _add_root_dir(self, path):
        path = path.split(",")
        for idx, p in enumerate(path):
            path[idx] = os.path.join(self.config.data_dir, p)

        return ",".join(path)
