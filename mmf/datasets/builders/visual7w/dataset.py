# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from torchvision.ops import box_iou


class Visual7WDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="visual7w", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)

    def __getitem__(self, idx):

        sample_info = self.annotation_db[idx]
        features_data = self.features_db[idx]
        multiple_choice_idx = torch.tensor(sample_info["mc_idx"])

        boxes = torch.from_numpy(features_data["image_info_0"]["bbox"])
        features = features_data["image_feature_0"]

        num_boxes = features.size()[0]
        global_feature = torch.sum(features, 0) / num_boxes
        global_feature = global_feature.unsqueeze(0)
        image_w = features_data["image_info_0"]["image_width"]
        image_h = features_data["image_info_0"]["image_height"]

        global_bounding_box = torch.tensor(
            [[0, 0, image_w, image_h]], dtype=torch.float
        )

        features = torch.cat((global_feature, features), 0)
        boxes = torch.cat((global_bounding_box, boxes), 0)
        num_boxes = num_boxes + 1

        gt_num_boxes = features_data["image_info_1"]["num_boxes"]
        gt_boxes = torch.from_numpy(
            features_data["image_info_1"]["bbox"][:gt_num_boxes, :]
        )
        gt_features = features_data["image_feature_1"][:gt_num_boxes, :]

        all_boxes = torch.cat((boxes, gt_boxes), 0)
        all_features = torch.cat((features, gt_features), 0)
        total_num_boxes = min(
            int(num_boxes + int(gt_num_boxes)), self.config.max_region_num
        )

        ref_box = sample_info["refBox"]
        # given the mix boxes, and ref_box, calculate the overlap.
        targets = box_iou(
            torch.tensor(all_boxes[:, :4]).float(), torch.tensor([ref_box]).float()
        )
        targets[targets < 0.5] = 0
        total_features = self.config.max_features + 1
        targets = targets[total_features:]
        targets = targets[multiple_choice_idx].squeeze()
        all_boxes_pad = torch.zeros((self.config.max_region_num, 4))
        all_features_pad = torch.zeros(
            (self.config.max_region_num, 2048), dtype=torch.float32
        )

        all_boxes_pad[:total_num_boxes] = all_boxes[:total_num_boxes]
        all_features_pad[:total_num_boxes] = all_features[:total_num_boxes]

        text_processor_argument = {"text": sample_info["caption"]}
        processed_question = self.text_processor(text_processor_argument)

        task_tokens = (
            processed_question["input_ids"]
            .new()
            .resize_(processed_question["input_ids"].size(0), 1)
            .fill_(int(4))
        )

        current_sample = Sample()
        current_sample.image_feature_0 = all_features_pad
        current_sample.input_ids = processed_question["input_ids"]
        current_sample.input_mask = processed_question["input_mask"]
        current_sample.segment_ids = processed_question["segment_ids"]
        current_sample.image_info_0 = {}
        current_sample.image_info_0["image_width"] = features_data["image_info_0"][
            "image_width"
        ]
        current_sample.image_info_0["image_height"] = features_data["image_info_0"][
            "image_height"
        ]
        current_sample.image_info_0["bbox"] = all_boxes_pad
        current_sample.image_info_0["max_features"] = torch.tensor(total_num_boxes)
        current_sample.image_info_0["max_gt_features"] = gt_num_boxes
        current_sample.image_info_0["max_image_features"] = num_boxes
        current_sample.image_info_0["multiple_choice_idx"] = multiple_choice_idx
        current_sample.targets = targets
        current_sample.task_tokens = task_tokens

        return current_sample

