# Copyright (c) Facebook, Inc. and its affiliates.
from multiprocessing.pool import ThreadPool

import torch
import tqdm

from pythia.common.registry import registry
from pythia.datasets.feature_readers import FeatureReader
from pythia.utils.distributed_utils import is_main_process


class FeaturesDataset:
    def __init__(self, features_type, *args, **kwargs):
        self.features_db = None
        if features_type == "coco":
            self.features_db = COCOFeaturesDataset(*args, **kwargs)
        else:
            raise ValueError("Unknown features' type {}".format(features_type))

    def __getattr__(self, name):
        if hasattr(self.features_db, name):
            return getattr(self.features_db, name)
        elif name in dir(self):
            return getattr(self, name)
        else:
            raise AttributeError(name)

    def __getitem__(self, idx):
        return self.features_db[idx]

    def __len__(self):
        return len(self.features_db)


class BaseFeaturesDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(BaseFeaturesDataset, self).__init__()


class COCOFeaturesDataset(BaseFeaturesDataset):
    def __init__(self, *args, **kwargs):
        super(COCOFeaturesDataset, self).__init__()
        self.feature_readers = []
        self.feature_dict = {}

        self.fast_read = kwargs["fast_read"]
        self.writer = registry.get("writer")

        for image_feature_dir in kwargs["directories"]:
            feature_reader = FeatureReader(
                base_path=image_feature_dir,
                depth_first=kwargs["depth_first"],
                max_features=kwargs["max_features"],
            )
            self.feature_readers.append(feature_reader)

        self.imdb = kwargs["imdb"]
        self.kwargs = kwargs
        self.should_return_info = kwargs.get("return_info", True)

        if self.fast_read:
            self.writer.write(
                "Fast reading features from %s" % (", ".join(kwargs["directories"]))
            )
            self.writer.write("Hold tight, this may take a while...")
            self._threaded_read()

    def _threaded_read(self):
        elements = [idx for idx in range(1, len(self.imdb))]
        pool = ThreadPool(processes=4)

        with tqdm.tqdm(total=len(elements), disable=not is_main_process()) as pbar:
            for i, _ in enumerate(pool.imap_unordered(self._fill_cache, elements)):
                if i % 100 == 0:
                    pbar.update(100)
        pool.close()

    def _fill_cache(self, idx):
        feat_file = self.imdb[idx]["feature_path"]
        features, info = self._read_features_and_info(feat_file)
        self.feature_dict[feat_file] = (features, info)

    def _read_features_and_info(self, feat_file):
        features = []
        infos = []
        for feature_reader in self.feature_readers:
            feature, info = feature_reader.read(feat_file)
            # feature = torch.from_numpy(feature).share_memory_()

            features.append(feature)
            infos.append(info)

        if not self.should_return_info:
            infos = None
        return features, infos

    def _get_image_features_and_info(self, feat_file):
        image_feats, infos = self.feature_dict.get(feat_file, (None, None))

        if image_feats is None:
            image_feats, infos = self._read_features_and_info(feat_file)

        # TODO: Remove after standardization
        # https://github.com/facebookresearch/pythia/blob/master/dataset_utils/dataSet.py#L226
        return image_feats, infos

    def __len__(self):
        return len(self.imdb) - 1

    def __getitem__(self, idx):
        image_info = self.imdb[idx]
        image_file_name = image_info.get("feature_path", None)

        if image_file_name is None:
            image_file_name = "{}.npy".format(image_info["image_id"])

        image_features, infos = self._get_image_features_and_info(image_file_name)

        item = {}
        for idx, image_feature in enumerate(image_features):
            item["image_feature_%s" % idx] = image_feature
            if infos is not None:
                item["image_info_%s" % idx] = infos[idx]

        return item
