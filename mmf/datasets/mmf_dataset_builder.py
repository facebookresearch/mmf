import collections
import os
import typing
import warnings

import mmf.utils.download as download
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.datasets.concat_dataset import MMFConcatDataset
from mmf.utils.configuration import get_global_config, get_zoo_config
from mmf.utils.general import get_absolute_path

ResourcesType = typing.NewType(
    "ResourcesType",
    typing.Union[
        typing.Mapping[
            str,
            typing.Union[
                download.DownloadableFile, typing.List[download.DownloadableFile]
            ],
        ],
        download.DownloadableFile,
        typing.Sequence[download.DownloadableFile],
        None,
    ],
)


class MMFDatasetBuilder(BaseDatasetBuilder):
    VERSION = None
    RESOURCES: ResourcesType = None
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
    def version(self):
        if self.VERSION is None:
            version, resources = get_zoo_config(
                self.dataset_name,
                self.zoo_variation,
                self.zoo_config_path,
                self.zoo_type,
            )
            self.VERSION = version
            self.RESOURCES = resources

            if version is None:
                # TODO: Convert this later to NotImplementedError once
                # all datasets have been migrated.
                warnings.warn(
                    "Dataset builder must define a version as class attribute "
                    + "'VERSION' or in dataset configuration."
                )
        return self.VERSION

    @version.setter
    def version(self, x):
        # Shouldn't be used, but just in case if needed
        self.VERSION = x

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

    @property
    def resources(self):
        if self.RESOURCES is None:
            version, resources = get_zoo_config(
                self.dataset_name,
                self.zoo_variation,
                self.zoo_config_path,
                self.zoo_type,
            )
            self.VERSION = version
            self.RESOURCES = resources
            if resources is None:
                warnings.warn(
                    "'RESOURCES' classes property has not been defined for the dataset "
                    + "builder. Nothing will be downloaded. "
                    + "Set 'RESOURCES' if you want to download or defined your "
                    + "zoo config in configs/zoo/datasets.yaml."
                )
                return {"features": [], "annotations": [], "images": [], "extras": []}
        return self.RESOURCES

    @resources.setter
    def resources(self, resources: ResourcesType):
        self.RESOURCES = resources

    def set_dataset_class(self, dataset_cls):
        self.dataset_class = dataset_cls

    def build(self, config, dataset_type="train", *args, **kwargs):
        resources: ResourcesType = self.resources

        if resources is None:
            return

        download_path = os.path.join(
            config.data_root_dir, "datasets", self.dataset_name
        )
        download_path = get_absolute_path(download_path)

        if not isinstance(resources, collections.abc.Mapping):
            self._download_resources(resources, download_path)
        else:
            use_features = config.get("use_features", False)
            use_images = config.get("use_images", False)

            if use_features:
                self._download_based_on_attribute(resources, download_path, "features")

            if use_images:
                self._download_based_on_attribute(resources, download_path, "images")

            self._download_based_on_attribute(resources, download_path, "annotations")
            self._download_resources(resources.get("extras", []), download_path)

    def load(self, config, dataset_type, *args, **kwargs):
        self.config = config
        annotations = config.get("annotations", {}).get(dataset_type, [])

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

    def _download_based_on_attribute(self, resources, download_path, attribute):
        path = os.path.join(download_path, attribute)
        self._download_resources(resources.get(attribute, []), path)

    def _download_resources(self, resources, path):
        download.download_resources(resources, path, self.version)
