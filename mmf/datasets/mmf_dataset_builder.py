import collections
import os
import typing
import warnings
from copy import deepcopy

import mmf.utils.download as download
import torch
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.datasets.concat_dataset import MMFConcatDataset
from mmf.datasets.subset_dataset import MMFSubset
from mmf.utils.configuration import get_global_config, get_mmf_env, get_zoo_config
from mmf.utils.general import get_absolute_path
from omegaconf import open_dict


class MMFDatasetBuilder(BaseDatasetBuilder):
    ZOO_CONFIG_PATH = None
    ZOO_VARIATION = None

    def __init__(
        self,
        dataset_name,
        dataset_class=None,
        zoo_variation="defaults",
        *args,
        **kwargs,
    ):
        super().__init__(dataset_name)
        self.dataset_class = dataset_class
        self.zoo_type = "datasets"
        self.zoo_variation = zoo_variation

    @property
    def dataset_class(self):
        return self._dataset_class

    @dataset_class.setter
    def dataset_class(self, dataset_class):
        self._dataset_class = dataset_class

    @property
    def zoo_variation(self):
        return self._zoo_variation

    @zoo_variation.setter
    def zoo_variation(self, zoo_variation):
        self._zoo_variation = zoo_variation

    @property
    def zoo_config_path(self):
        if self.ZOO_CONFIG_PATH is None:
            self.ZOO_CONFIG_PATH = get_global_config("env.dataset_zoo")
        return self.ZOO_CONFIG_PATH

    @zoo_config_path.setter
    def zoo_config_path(self, zoo_config_path):
        self.ZOO_CONFIG_PATH = zoo_config_path

    def set_dataset_class(self, dataset_cls):
        self.dataset_class = dataset_cls

    def build(self, config, dataset_type="train", *args, **kwargs):
        requirements = config.get("zoo_requirements", [])

        if len(requirements) == 0:
            # If nothing is specified, build the default requirement
            self._download_requirement(config, self.dataset_name, self.zoo_variation)
        else:
            # Else build all of the requirements one by one
            # Default must also be specified in these requirements if needed
            for requirement in requirements:
                self._download_requirement(config, requirement)

    def _download_requirement(
        self, config, requirement_key, requirement_variation="defaults"
    ):
        version, resources = get_zoo_config(
            requirement_key, requirement_variation, self.zoo_config_path, self.zoo_type
        )

        if resources is None:
            return

        requirement_split = requirement_key.split(".")
        dataset_name = requirement_split[0]

        # The dataset variation has been directly passed in the key so use it instead
        if len(requirement_split) >= 2:
            dataset_variation = requirement_split[1]
        else:
            dataset_variation = requirement_variation

        # We want to use root env data_dir so that we don't mix up our download
        # root dir with the dataset ones
        download_path = os.path.join(
            get_mmf_env("data_dir"), "datasets", dataset_name, dataset_variation
        )
        download_path = get_absolute_path(download_path)

        if not isinstance(resources, collections.abc.Mapping):
            self._download_resources(resources, download_path, version)
        else:
            use_features = config.get("use_features", False)
            use_images = config.get("use_images", False)

            if use_features:
                self._download_based_on_attribute(
                    resources, download_path, version, "features"
                )

            if use_images:
                self._download_based_on_attribute(
                    resources, download_path, version, "images"
                )

            self._download_based_on_attribute(
                resources, download_path, version, "annotations"
            )
            self._download_resources(
                resources.get("extras", []), download_path, version
            )

    def load(self, config, dataset_type, *args, **kwargs):
        self.config = config

        split_dataset_from_train = self.config.get("split_train", False)
        if split_dataset_from_train:
            config = self._modify_dataset_config_for_split(config)

        annotations = self._read_annotations(config, dataset_type)
        if annotations is None:
            return None

        datasets = []
        for imdb_idx in range(len(annotations)):
            dataset_class = self.dataset_class
            dataset = dataset_class(config, dataset_type, imdb_idx)
            datasets.append(dataset)

        dataset = MMFConcatDataset(datasets)
        if split_dataset_from_train:
            dataset = self._split_dataset_from_train(dataset, dataset_type)

        self.dataset = dataset
        return self.dataset

    def _split_dataset_from_train(self, dataset, dataset_type):
        if dataset_type in self.config.split_train.keys() or dataset_type == "train":
            start, end = self._calculate_split_for_dataset_type(dataset_type)
            dataset_length = len(dataset)
            start, end = round(start * dataset_length), round(end * dataset_length)
            if start > end:
                raise ValueError(
                    f"Train split ratio for {dataset_type} must be positive."
                )
            indices = self._generate_permuted_indexes(dataset_length)[start:end]
            dataset = MMFSubset(dataset, indices)
            print(
                f"Dataset type: {dataset_type} length: {len(dataset)} total: {dataset_length}"
            )
        return dataset

    def _generate_permuted_indexes(self, dataset_length):
        generator = torch.Generator()
        generator.manual_seed(self.config.get("split_train.seed", 123456))
        return torch.randperm(dataset_length, generator=generator)

    def _modify_dataset_config_for_split(self, config):
        with open_dict(config):
            for data_type in config.split_train:
                if data_type == "seed":
                    continue
                if config.use_images:
                    config.images[data_type] = deepcopy(config.images.train)
                if config.use_features:
                    config.features[data_type] = deepcopy(config.features.train)
                config.annotations[data_type] = deepcopy(config.annotations.train)
        return config

    def _read_annotations(self, config, dataset_type):
        annotations = config.get("annotations", {}).get(dataset_type, [])

        # User can pass a single string as well
        if isinstance(annotations, str):
            annotations = [annotations]

        if len(annotations) == 0:
            warnings.warn(
                "Dataset type {} is not present or empty in "
                + "annotations of dataset config or either annotations "
                + "key is not present. Returning None. "
                + "This dataset won't be used.".format(dataset_type)
            )
            return None
        return annotations

    def _calculate_split_for_dataset_type(self, dataset_type):
        start = 0.0
        for data_type in self.config.split_train:
            if data_type == "seed":
                continue
            if dataset_type == data_type:
                return (start, start + self.config.split_train[data_type])
            start += self.config.split_train[data_type]

        if start > 1.0:
            raise ValueError(
                "Ratios of val plus test should not exceed 100%."
                + " Need to leave some percentage for training."
            )
        elif start == 1.0:
            warnings.warn("All data in training set is used for val and/or test.")

        if dataset_type == "train":
            return (start, 1.0)

    def _download_based_on_attribute(
        self, resources, download_path, version, attribute
    ):
        path = os.path.join(download_path, attribute)
        self._download_resources(resources.get(attribute, []), path, version)

    def _download_resources(self, resources, path, version):
        download.download_resources(resources, path, version)
