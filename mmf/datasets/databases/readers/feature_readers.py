# Copyright (c) Facebook, Inc. and its affiliates.

import math
import os
import pickle
from typing import Any

import lmdb
import numpy as np
import torch
from mmf.utils.file_io import PathManager


def load_feat(feat_path: str, convert_to_tensor: bool = False) -> Any:
    with PathManager.open(feat_path, "rb") as f:
        if feat_path.endswith("npy"):
            feat = np.load(f, allow_pickle=True)
            if convert_to_tensor:
                feat = torch.from_numpy(feat)
        elif feat_path.endswith("pth"):
            feat = torch.load(f, map_location=torch.device("cpu"))
        else:
            raise AssertionError("Unknown feature type")

    return feat


class FeatureReader:
    def __init__(self, base_path, depth_first, max_features=None):
        """Feature Reader class for reading features.

        Note: Deprecation: ndim and image_feature will be deprecated later
        and the format will be standardize using features from detectron.

        Parameters
        ----------
        ndim : int
            Number of expected dimensions in features
        depth_first : bool
            CHW vs HWC
        max_features : int
            Number of maximum bboxes to keep

        Returns
        -------
        type
            Description of returned object.

        """
        self.base_path = base_path
        ndim = None
        self.feat_reader = None
        self.depth_first = depth_first
        self.max_features = max_features
        self.ndim = ndim

    def _init_reader(self):
        # Currently all lmdb features are with ndim == 2
        if self.base_path.endswith(".lmdb"):
            self.feat_reader = LMDBFeatureReader(self.max_features, self.base_path)
        elif self.ndim == 2 or self.ndim == 0:
            if self.max_features is None:
                self.feat_reader = FasterRCNNFeatureReader()
            else:
                # TODO: Fix later when we move to proper standardized features
                # if isinstance(self.image_feature.item(0), dict):
                #     self.feat_reader = \
                #         PaddedFeatureRCNNWithBBoxesFeatureReader(
                #             self.max_features
                #         )
                # else:
                self.feat_reader = PaddedFasterRCNNFeatureReader(self.max_features)
        elif self.ndim == 3 and not self.depth_first:
            self.feat_reader = Dim3FeatureReader()
        elif self.ndim == 4 and self.depth_first:
            self.feat_reader = CHWFeatureReader(self.max_features)
        elif self.ndim == 4 and not self.depth_first:
            self.feat_reader = HWCFeatureReader()
        else:
            raise TypeError("unknown image feature format")

    def read(self, image_feat_path):
        if not image_feat_path.endswith("npy") and not image_feat_path.endswith("pth"):
            return None
        image_feat_path = os.path.join(self.base_path, image_feat_path)

        if self.feat_reader is None:
            # Currently all lmdb features are with ndim == 2 so we are
            # avoiding loading the lmdb to determine feature ndim
            if not self.base_path.endswith(".lmdb") and self.ndim is None:
                feat = load_feat(image_feat_path)
                self.ndim = feat.ndim
            self._init_reader()

        return self.feat_reader.read(image_feat_path)


class FasterRCNNFeatureReader:
    def read(self, image_feat_path):
        feat = load_feat(image_feat_path, convert_to_tensor=True)
        return feat, None


class CHWFeatureReader:
    def __init__(self, max_features=None):
        self.max_features = max_features
        if self.max_features:
            patch_dim = math.ceil(math.sqrt(self.max_features))
            self.img_h = patch_dim
            self.img_w = patch_dim

    def read(self, image_feat_path):
        feat = load_feat(image_feat_path, convert_to_tensor=True)
        assert feat.shape[0] == 1, "batch is not 1"
        b, c, h, w = feat.shape
        if self.max_features:
            padded_feat = torch.zeros((b, c, self.img_h, self.img_w), dtype=torch.float)
            padded_feat[:, :, :h, :w] = feat
            feat = padded_feat
        feat = feat.squeeze(0)
        return feat, None


class Dim3FeatureReader:
    def read(self, image_feat_path):
        tmp = load_feat(image_feat_path)
        _, _, c_dim = tmp.shape
        image_feature = torch.from_numpy(np.reshape(tmp, (-1, c_dim)))
        return image_feature, None


class HWCFeatureReader:
    def read(self, image_feat_path):
        tmp = load_feat(image_feat_path)
        assert tmp.shape[0] == 1, "batch is not 1"
        _, _, _, c_dim = tmp.shape
        image_feature = torch.from_numpy(np.reshape(tmp, (-1, c_dim)))
        return image_feature, None


class PaddedFasterRCNNFeatureReader:
    def __init__(self, max_loc):
        self.max_loc = max_loc
        self.first = True
        self.take_item = False

    def _load(self, image_feat_path):
        image_info = {}
        image_info["features"] = load_feat(image_feat_path)

        info_path = "{}_info.npy".format(image_feat_path.split(".npy")[0])
        if PathManager.exists(info_path):
            image_info.update(load_feat(info_path).item())

        return image_info

    def read(self, image_feat_path):
        image_info = self._load(image_feat_path)
        if self.first:
            self.first = False
            if (
                image_info["features"].size == 1
                and "image_feat" in image_info["features"].item()
            ):
                self.take_item = True

        image_feature = image_info["features"]

        if self.take_item:
            item = image_info["features"].item()
            if "image_text" in item:
                image_info["image_text"] = item["image_text"]
                image_info["is_ocr"] = item["image_bbox_source"]
                image_feature = item["image_feat"]

            if "info" in item:
                if "image_text" in item["info"]:
                    image_info.update(item["info"])
                image_feature = item["feature"]

        # Handle the case of ResNet152 features
        if len(image_feature.shape) > 2:
            shape = image_feature.shape
            image_feature = image_feature.reshape(-1, shape[-1])

        image_loc, image_dim = image_feature.shape
        tmp_image_feat = np.zeros((self.max_loc, image_dim), dtype=np.float32)
        tmp_image_feat[0:image_loc,] = image_feature[: self.max_loc, :]  # noqa
        image_feature = torch.from_numpy(tmp_image_feat)

        del image_info["features"]
        image_info["max_features"] = torch.tensor(image_loc, dtype=torch.long)
        return image_feature, image_info


class LMDBFeatureReader(PaddedFasterRCNNFeatureReader):
    def __init__(self, max_loc, base_path):
        super().__init__(max_loc)
        self.db_path = base_path

        if not PathManager.exists(self.db_path):
            raise RuntimeError(
                "{} path specified for LMDB features doesn't exists.".format(
                    self.db_path
                )
            )
        self.env = None

    def _init_db(self):
        self.env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False, buffers=True) as txn:
            self.image_ids = pickle.loads(txn.get(b"keys"))
            self.image_id_indices = {
                self.image_ids[i]: i for i in range(0, len(self.image_ids))
            }

    def _load(self, image_file_path):
        if self.env is None:
            self._init_db()

        split = os.path.relpath(image_file_path, self.db_path).split(".npy")[0]

        try:
            image_id = int(split.split("_")[-1])
            # Try fetching to see if it actually exists otherwise fall back to
            # default
            img_id_idx = self.image_id_indices[str(image_id).encode()]
        except (ValueError, KeyError):
            # The image id is complex or involves folder, use it directly
            image_id = str(split).encode()
            img_id_idx = self.image_id_indices[image_id]

        with self.env.begin(write=False, buffers=True) as txn:
            image_info = pickle.loads(txn.get(self.image_ids[img_id_idx]))

        return image_info


class PaddedFeatureRCNNWithBBoxesFeatureReader:
    def __init__(self, max_loc):
        self.max_loc = max_loc

    def read(self, image_feat_path):
        image_feat_bbox = load_feat(image_feat_path)
        image_boxes = image_feat_bbox.item().get("image_bboxes")
        tmp_image_feat = image_feat_bbox.item().get("image_feature")
        image_loc, image_dim = tmp_image_feat.shape
        tmp_image_feat_2 = np.zeros((self.max_loc, image_dim), dtype=np.float32)
        tmp_image_feat_2[0:image_loc,] = tmp_image_feat  # noqa
        tmp_image_feat_2 = torch.from_numpy(tmp_image_feat_2)
        tmp_image_box = np.zeros((self.max_loc, 4), dtype=np.int32)
        tmp_image_box[0:image_loc] = image_boxes
        tmp_image_box = torch.from_numpy(tmp_image_box)
        image_info = {
            "image_bbox": tmp_image_box,
            "max_features": torch.tensor(image_loc, dtype=torch.int),
        }

        return tmp_image_feat_2, image_info
