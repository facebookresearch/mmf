# Requires vqa-maskrcnn-benchmark to be built and installed
# Category mapping for visual genome can be downloaded from
# https://dl.fbaipublicfiles.com/pythia/data/visual_genome_categories.json
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from pythia.utils.general import download_file


class FeatureExtractor:
    MODEL_URL = (
        "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth"
    )
    CONFIG_URL = (
        "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml"
    )
    MAX_SIZE = 1333
    MIN_SIZE = 800
    NUM_FEATURES = 100

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.detection_model = self._build_detection_model()

        os.makedirs(self.args.output_folder, exist_ok=True)

    def _try_downloading_necessities(self):
        if self.args.model_file is None:
            print("Downloading model and configuration")
            self.args.model_file = self.MODEL_URL.split("/")[-1]
            self.args.config_file = self.CONFIG_URL.split("/")[-1]
            download_file(self.MODEL_URL)
            download_file(self.CONFIG_URL)

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_file", default=None, type=str, help="Detectron model file"
        )
        parser.add_argument(
            "--config_file", default=None, type=str, help="Detectron config file"
        )
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument(
            "--output_folder", type=str, default="./output", help="Output folder"
        )
        parser.add_argument("--image_dir", type=str, help="Image directory or file")
        parser.add_argument(
            "--feature_name", type=str, help="The name of the feature to extract",
            default="fc6",
        )
        parser.add_argument(
            "--confidence_threshold", type=float, default=0.2,
            help="Threshold of detection confidence above which boxes will be selected"
        )
        return parser

    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)
        return img, im_scale

    def _process_feature_extraction(
        self, output, im_scales, feature_name="fc6", conf_thresh=0.2
    ):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)

            for cls_ind in range(1, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep]
                )

            keep_boxes = torch.argsort(max_conf, descending=True)[:self.NUM_FEATURES]
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            objects = torch.argmax(scores[keep_boxes], dim=1)
            image_width = output[0]["proposals"][i].size[0] / im_scales[i]
            image_height = output[0]["proposals"][i].size[1] / im_scales[i]

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "objects": objects.cpu().numpy(),
                    "image_width": image_width,
                    "image_height": image_height,
                }
            )

        return feat_list, info_list

    def get_detectron_features(self, image_paths):
        img_tensor, im_scales = [], []

        for image_path in image_paths:
            im, im_scale = self._image_transform(image_path)
            img_tensor.append(im)
            im_scales.append(im_scale)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to("cuda")

        with torch.no_grad():
            output = self.detection_model(current_img_list)
        feat_list = self._process_feature_extraction(
            output, im_scales, self.args.feature_name, self.args.confidence_threshold
        )
        return feat_list

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info_file_base_name = file_base_name + "_info.npy"
        file_base_name = file_base_name + ".npy"

        np.save(
            os.path.join(self.args.output_folder, file_base_name), feature.cpu().numpy()
        )
        np.save(os.path.join(self.args.output_folder, info_file_base_name), info)

    def extract_features(self):
        image_dir = self.args.image_dir

        if os.path.isfile(image_dir):
            features, infos = self.get_detectron_features([image_dir])
            self._save_feature(image_dir, features[0], infos[0])
        else:
            files = glob.glob(os.path.join(image_dir, "*.jpg"))
            for chunk in self._chunks(files, self.args.batch_size):
                features, infos = self.get_detectron_features(chunk)
                for idx, file_name in enumerate(chunk):
                    self._save_feature(file_name, features[idx], infos[idx])


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
