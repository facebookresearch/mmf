# Copyright (c) Facebook, Inc. and its affiliates.

# This script extracts features for vinvl using their scene-graph-benchmark repo.
# Their AttrRCNN model can not support JIT, so we need clone the repo for
# their pytorch code. This script assumes you followed the vinvl feature
# extraction tutorial! The setup can be straight forward.
# This script is a modification of their
# tools/demo/demo_image.py for feature extraction for image directories
# as suggested by SPQRXVIII001 in the github issues.
# If you would like vinvl features for a popular dataset, consider
# downloading the pre-extracted features from
# https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md
# This script was written for scene_graph_branchmark commit version 8e14944
# oscar (vinvl repo) version 4788a74


import argparse
import os

import cv2
import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from mmf.utils.download import download
from PIL import Image
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from scene_graph_benchmark.config import sg_cfg
from tools.scripts.features.extraction_utils import chunks, get_image_files


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureExtractor:
    MODEL_URL = {
        "X-152-C4": "https://dl.fbaipublicfiles.com/mmf/data/models/vinvl/"
        + "detection/vinvl_vg_x152c4.pth",
    }
    CONFIG_URL = {
        "X-152-C4": "https://dl.fbaipublicfiles.com/mmf/data/models/vinvl/"
        + "detection/vinvl_x152c4.yaml",
    }

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self._try_downloading_necessities(self.args.model_name)
        self.detection_model = self._build_detection_model()
        self.transforms = build_transforms(cfg, is_train=False)

        os.makedirs(self.args.output_folder, exist_ok=True)

    def _try_downloading_necessities(self, model_name):
        if self.args.model_file is None and model_name is not None:
            model_url = self.MODEL_URL[model_name]
            config_url = self.CONFIG_URL[model_name]
            self.args.model_file = model_url.split("/")[-1]
            self.args.config_file = config_url.split("/")[-1]
            if os.path.exists(self.args.model_file) and os.path.exists(
                self.args.config_file
            ):
                print(f"model and config file exists in directory: {os.getcwd()}")
                return
            print("Downloading model and configuration")
            download(model_url, ".", self.args.model_file)
            download(config_url, ".", self.args.config_file)

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_name",
            default="X-152-C4",
            type=str,
            help="Model to use for detection",
        )
        parser.add_argument(
            "--model_file",
            default=None,
            type=str,
            help="Detectron model file. This overrides the model_name param.",
        )
        parser.add_argument(
            "--config_file", default=None, type=str, help="Detectron config file"
        )
        parser.add_argument(
            "--start_index", default=0, type=int, help="Index to start from "
        )
        parser.add_argument("--end_index", default=None, type=int, help="")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--output_folder", type=str, default="./output", help="Output folder"
        )
        parser.add_argument("--image_dir", type=str, help="Image directory or file")
        parser.add_argument(
            "--exclude_list",
            type=str,
            help="List of images to be excluded from feature conversion. "
            + "Each image on a new line",
            default="./list",
        )
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        return parser

    def _build_detection_model(self):
        """Create model cfg for feature extraction
        SG expects a yacs config (CfgNode)
        which is effectively a dictionary merged from
        their cfg defaults, their sg_cfg defaults, and
        scene_graph_benchmark/sgg_configs/vgattr/vinvl_x152c4.yaml
        the last needs to be modified for vinvl feature extraction
        we will modify that cfg obj here to output box_features
        used by oscar (vinvl model) as image feature input
        """
        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.set_new_allowed(False)
        # Configuring VinVl
        cfg.merge_from_file(self.args.config_file)
        model_args = [
            "MODEL.WEIGHT",
            self.args.model_file,
            "MODEL.ROI_HEADS.NMS_FILTER",
            1,
            "MODEL.ROI_HEADS.SCORE_THRESH",
            0.2,
            "TEST.IGNORE_BOX_REGRESSION",
            False,
            "MODEL.ATTRIBUTE_ON",
            True,
            "TEST.OUTPUT_FEATURE",
            True,
            "TEST.OUTPUT_RELATION_FEATURE",
            True,
            "TEST.TSV_SAVE_SUBSET",
            ["rect", "class", "conf", "feature", "relation_feature"],
            "TEST.GATHER_ON_CPU",
            True,
        ]
        cfg.merge_from_list(model_args)
        cfg.freeze()

        model = AttrRCNN(cfg)
        model.to(DEVICE)
        model.eval()
        output_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
        checkpointer.load(cfg.MODEL.WEIGHT)
        return model

    def _image_transform(self, path):
        img = cv2.imread(path)
        img_height = img.shape[0]
        img_width = img.shape[1]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img, _ = self.transforms(img, target=None)
        img = img.to(DEVICE)
        return img, {"width": img_width, "height": img_height}

    def _norm_box(self, bbox, w, h):
        bbox_aug = torch.zeros(bbox.size(0), 6)
        bbox_aug[:, :4] = bbox
        bbox_aug[:, 0] /= w
        bbox_aug[:, 1] /= h
        bbox_aug[:, 2] /= w
        bbox_aug[:, 3] /= h
        bbox_aug[:, 4] = bbox_aug[:, 2] - bbox_aug[:, 0]
        bbox_aug[:, 5] = bbox_aug[:, 3] - bbox_aug[:, 1]
        return bbox_aug

    def _process_feature_extraction(self, output, im_infos):
        """Convert AttrRCNN object detection output and image sizes
        to make image features and image info expected by MMF.

        Args:
            output (List[BoxList]):
                A list of OD outputs, with BoxList containing image_info and feats
                BoxList fields contain labels, scores, box_features,
                attr_labels, attr_scores based on config options.
                bbox attribute contains bounding boxes.

            im_infos (List[Dict[str,int]]):
                A list of Dicts containing image width, height values.

        Returns:
            Tuple[List[np.Array], List[Dict[str, Union[torch.Tensor, int]]]]:
                Returns a list of image features, and list of image info dicts
        """
        feat_list = []
        info_list = []

        for i, box_list in enumerate(output):
            w = im_infos[i]["width"]
            h = im_infos[i]["height"]
            box_list = box_list.to("cpu").resize((w, h))
            det_dict = {key: box_list.get_field(key) for key in box_list.fields()}
            bbox = box_list.bbox
            bbox_aug = self._norm_box(bbox, w, h)
            det_dict["bbox"] = bbox_aug
            det_dict["image_width"] = w
            det_dict["image_height"] = h
            features = torch.cat([det_dict["box_features"], det_dict["bbox"]], dim=1)

            det_dict = {
                key: val.numpy() if isinstance(val, torch.Tensor) else val
                for key, val in det_dict.items()
            }
            features = features.numpy()
            feat_list += [features]
            info_list += [det_dict]

        return feat_list, info_list

    def get_vinvl_features(self, image_paths):
        img_tensor, im_infos = [], []

        for image_path in image_paths:
            im, im_info = self._image_transform(image_path)
            img_tensor.append(im)
            im_infos.append(im_info)

        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to(DEVICE)

        torch.manual_seed(0)
        with torch.no_grad():
            output = self.detection_model(current_img_list)

        feat_list = self._process_feature_extraction(
            output,
            im_infos,
        )
        return feat_list

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info_file_base_name = file_base_name + "_info.npy"
        file_base_name = file_base_name + ".npy"

        np.save(os.path.join(self.args.output_folder, file_base_name), feature)
        np.save(os.path.join(self.args.output_folder, info_file_base_name), info)

    def extract_features(self):
        """Models and config files are downloaded if not
        specified in args.
        Then gets a list of images to extract features from.
        Will exclude images already in the output dir,
        and images in the exclude list.
        Images are loaded and transformed based on config.
        Will then do feature extraction in batches
        using AttrRCNN with weights from scene graph benchmark
        for VinVL (Oscar+) image features.
        Output of a list of BoxList objects that contain
        fields and bbox which are processes to create
        image features and image info objects for MMF.

        By default this will generate npy files containing
        the image features expected by VinVL checkpoints,
        and work with MMF out of the box.
        """
        image_dir = self.args.image_dir
        if os.path.isfile(image_dir):
            features, infos = self.get_vinvl_features([image_dir])
            self._save_feature(image_dir, features[0], infos[0])
        else:
            files = get_image_files(
                self.args.image_dir,
                exclude_list=self.args.exclude_list,
                start_index=self.args.start_index,
                end_index=self.args.end_index,
                output_folder=self.args.output_folder,
            )

            finished = 0
            total = len(files)

            for chunk, _ in chunks(files, self.args.batch_size):
                features, infos = self.get_vinvl_features(chunk)
                for idx, file_name in enumerate(chunk):
                    self._save_feature(file_name, features[idx], infos[idx])
                finished += len(chunk)

                if finished % 200 == 0:
                    print(f"Processed {finished}/{total}")


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
