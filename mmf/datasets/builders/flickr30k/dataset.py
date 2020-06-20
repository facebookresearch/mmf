# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np

from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset


class Flickr30kDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="visual7w", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)

    def __getitem__(self, idx):

        sample_info = self.annotation_db[idx]
        features_data = self.features_db[idx]


        
        multiple_choice_idx = torch.from_numpy(np.array(sample_info["mc_idx"]))

        boxes = features_data['image_info_0']['bbox']
        gt_boxes = \
            features_data['image_info_1']['bbox'][multiple_choice_idx, :]
        features = features_data['image_feature_0']
        gt_features = \
            features_data['image_feature_1'][multiple_choice_idx, :]

        mix_boxes = np.concatenate((boxes, gt_boxes), axis=0)
        mix_features = torch.tensor(np.concatenate((features, gt_features), axis=0))
        mix_max_features = features_data['image_info_0']['max_features'] \
            + features_data['image_info_1']['max_features']

        text_processor_argument = {"text": sample_info["caption"]}
        processed_question = self.text_processor(text_processor_argument)

        ref_box = sample_info["refBox"]

        # given the mix boxes, and ref_box, calculate the overlap.
        targets = self.iou(
            torch.tensor(gt_boxes[:, :4]).float(), torch.tensor([ref_box]).float()
        ).numpy()
        targets[targets < 0.5] = 0

        targets = torch.tensor(targets)
        targets = targets.contiguous().view(-1)
        #targets = np.argmax(targets)

        current_sample = Sample()
        current_sample.image_feature_0 = mix_features
        current_sample.input_ids = processed_question["input_ids"]
        current_sample.input_mask = processed_question["input_mask"]
        current_sample.segment_ids = processed_question["segment_ids"]
        current_sample.image_info_0 = {}
        current_sample.image_info_0["image_width"] = \
            features_data['image_info_0']['image_width']
        current_sample.image_info_0["image_height"] = \
            features_data['image_info_0']['image_height']
        current_sample.image_info_0['bbox'] = mix_boxes
        current_sample.image_info_0['max_features'] = mix_max_features
        current_sample.targets = targets
        return current_sample
