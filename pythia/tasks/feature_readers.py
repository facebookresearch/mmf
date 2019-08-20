# Copyright (c) Facebook, Inc. and its affiliates.
import os

import numpy as np
import torch


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
        if self.ndim == 2 or self.ndim == 0:
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
            self.feat_reader = CHWFeatureReader()
        elif self.ndim == 4 and not self.depth_first:
            self.feat_reader = HWCFeatureReader()
        else:
            raise TypeError("unkown image feature format")

    def read(self, image_feat_path):
        if not image_feat_path.endswith("npy"):
            return None
        image_feat_path = os.path.join(self.base_path, image_feat_path)

        if self.feat_reader is None:
            if self.ndim is None:
                feat = np.load(image_feat_path)
                self.ndim = feat.ndim
            self._init_reader()

        return self.feat_reader.read(image_feat_path)


class FasterRCNNFeatureReader:
    def read(self, image_feat_path):
        return torch.from_numpy(np.load(image_feat_path)), None


class CHWFeatureReader:
    def read(self, image_feat_path):
        feat = np.load(image_feat_path)
        assert feat.shape[0] == 1, "batch is not 1"
        feat = torch.from_numpy(feat.squeeze(0))
        return feat, None


class Dim3FeatureReader:
    def read(self, image_feat_path):
        tmp = np.load(image_feat_path)
        _, _, c_dim = tmp.shape
        image_feature = torch.from_numpy(np.reshape(tmp, (-1, c_dim)))
        return image_feature, None


class HWCFeatureReader:
    def read(self, image_feat_path):
        tmp = np.load(image_feat_path)
        assert tmp.shape[0] == 1, "batch is not 1"
        _, _, _, c_dim = tmp.shape
        image_feature = torch.from_numpy(np.reshape(tmp, (-1, c_dim)))
        return image_feature, None


class PaddedFasterRCNNFeatureReader:
    def __init__(self, max_loc):
        self.max_loc = max_loc
        self.first = True
        self.take_item = False

    def read(self, image_feat_path):
        content = np.load(image_feat_path, allow_pickle=True)
        info_path = "{}_info.npy".format(image_feat_path.split(".npy")[0])
        image_info = {}

        if os.path.exists(info_path):
            image_info.update(np.load(info_path, allow_pickle=True).item())

        if self.first:
            self.first = False
            if content.size == 1 and "image_feat" in content.item():
                self.take_item = True

        image_feature = content

        if self.take_item:
            item = content.item()
            if "image_text" in item:
                image_info["image_text"] = item["image_text"]
                image_info["is_ocr"] = item["image_bbox_source"]
                image_feature = item["image_feat"]

            if "info" in item:
                if "image_text" in item["info"]:
                    image_info.update(item["info"])
                image_feature = item["feature"]

        image_loc, image_dim = image_feature.shape
        tmp_image_feat = np.zeros((self.max_loc, image_dim), dtype=np.float32)
        tmp_image_feat[0:image_loc,] = image_feature
        image_feature = torch.from_numpy(tmp_image_feat)

        image_info["max_features"] = torch.tensor(image_loc, dtype=torch.long)
        return image_feature, image_info


class PaddedFeatureRCNNWithBBoxesFeatureReader:
    def __init__(self, max_loc):
        self.max_loc = max_loc

    def read(self, image_feat_path):
        image_feat_bbox = np.load(image_feat_path)
        image_boxes = image_feat_bbox.item().get("image_bboxes")
        tmp_image_feat = image_feat_bbox.item().get("image_feature")
        image_loc, image_dim = tmp_image_feat.shape
        tmp_image_feat_2 = np.zeros((self.max_loc, image_dim), dtype=np.float32)
        tmp_image_feat_2[0:image_loc,] = tmp_image_feat
        tmp_image_feat_2 = torch.from_numpy(tmp_image_feat_2)
        tmp_image_box = np.zeros((self.max_loc, 4), dtype=np.int32)
        tmp_image_box[0:image_loc] = image_boxes
        tmp_image_box = torch.from_numpy(tmp_image_box)
        image_info = {
            "image_bbox": tmp_image_box,
            "max_features": torch.tensor(image_loc, dtype=torch.int),
        }

        return tmp_image_feat_2, image_info
