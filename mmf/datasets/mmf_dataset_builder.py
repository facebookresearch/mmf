import collections
import os
import typing
import warnings

import mmf.utils.download as download
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.datasets.concat_dataset import MMFConcatDataset
from mmf.utils.configuration import get_global_config, get_mmf_env, get_zoo_config
from mmf.utils.general import get_absolute_path


class MMFDatasetBuilder(BaseDatasetBuilder):
    ZOO_CONFIG_PATH = None
    ZOO_VARIATION = None

    def __init__(
        self,
        dataset_name,
        dataset_class=None,
        zoo_variation="defaults",
        *args,
        **kwargs
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

        datasets = []

        for imdb_idx in range(len(annotations)):
            dataset_class = self.dataset_class
            dataset = dataset_class(config, dataset_type, imdb_idx)
            datasets.append(dataset)

        dataset = MMFConcatDataset(datasets)
        self.dataset = dataset
        return self.dataset

    def _download_based_on_attribute(
        self, resources, download_path, version, attribute
    ):
        path = os.path.join(download_path, attribute)
        self._download_resources(resources.get(attribute, []), path, version)

    def _download_resources(self, resources, path, version):
        download.download_resources(resources, path, version)
