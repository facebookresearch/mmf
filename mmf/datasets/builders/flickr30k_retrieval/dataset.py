# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np

from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset


class Flickr30kRetrievalDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="flickr30k_retrieval", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)

        # +1 because old codebase adds a global feature while reading the features
        # from the lmdb. If we set config.max_features to + 1 value already in yaml,
        # this codebase adds zero padding. To work around that, setting #features to
        # max_region_num - 1, so this code doesnt add zero padding and then add global
        # feature and increment _max_region_num by one like below.
        self._max_region_num = self.config.max_features + 1

    def __getitem__(self, idx):

        sample_info = self.annotation_db[idx]

        # 1: correct one
        features1, mix_num_boxes1, spatials1, image_mask1 = \
            self.get_feature(sample_info[0]['image_id'])
        text_processor_argument = {"text": sample_info[0]["caption"]}
        processed_caption = self.text_processor(text_processor_argument)
        caption1 = processed_caption["input_ids"]
        input_mask1 = processed_caption["input_mask"]
        segment_ids1 = processed_caption["segment_ids"]

        #2: random caption wrong
        text_processor_argument = {"text": sample_info[1]["caption"]}
        processed_caption = self.text_processor(text_processor_argument)
        features2 = features1
        image_mask2 = image_mask1
        spatials2 = spatials1
        caption2 = processed_caption["input_ids"]
        input_mask2 = processed_caption["input_mask"]
        segment_ids2 = processed_caption["segment_ids"]

        #3: random image wrong
        features3, mix_num_boxes3, spatials3, image_mask3 = \
            self.get_feature(sample_info[2]['image_id'])
        caption3 = caption1
        input_mask3 = input_mask1
        segment_ids3 = segment_ids1

        #4: hard image wrong
        features4 = features1
        image_mask4 = image_mask1
        spatials4 = spatials1
        text_processor_argument = {"text": sample_info[3]["caption"]}
        processed_caption = self.text_processor(text_processor_argument)
        caption4 = processed_caption["input_ids"]
        input_mask4 = processed_caption["input_mask"]
        segment_ids4 = processed_caption["segment_ids"]

        features = torch.stack([features1, features2, features3, features4], dim=0)
        spatials = torch.stack([spatials1, spatials2, spatials3, spatials4], dim=0)
        image_mask = torch.stack(
            [image_mask1, image_mask2, image_mask3, image_mask4], dim=0
        )
        caption = torch.stack([caption1, caption2, caption3, caption4], dim=0)
        input_mask = torch.stack(
            [input_mask1, input_mask2, input_mask3, input_mask4], dim=0
        )
        segment_ids = torch.stack(
            [segment_ids1, segment_ids2, segment_ids3, segment_ids4], dim=0
        )
        target = torch.tensor(0).long()

        current_sample = Sample()
        current_sample.image_feature_0 = features
        current_sample.input_ids = caption
        current_sample.input_mask = input_mask
        current_sample.segment_ids = segment_ids
        current_sample.image_info_0 = {}
        current_sample.image_info_0['bbox'] = spatials
        current_sample.targets = target

        return current_sample

    def get_feature(self, image_id):

        image_id = str(image_id) + '.npy'
        features_data = self.features_db.from_path(image_id)

        boxes = features_data['image_info_0']['bbox']
        features = features_data['image_feature_0']

        # Adding global feature
        num_boxes = features.shape[0]
        g_feat = torch.sum(features, 0) / num_boxes
        g_feat = g_feat.unsqueeze(0)
        features = torch.cat(
            (g_feat, features), 0
        )

        # Adding global box
        image_w = features_data['image_info_0']['image_width']
        image_h = features_data['image_info_0']['image_height']
        g_location_ori = np.array(
            [[0, 0, image_w, image_h]]
        ).astype(np.float32)
        boxes = np.concatenate((g_location_ori, boxes), axis=0)
        num_boxes = num_boxes + 1

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]
        mix_boxes_pad[:mix_num_boxes, :4] = boxes[:mix_num_boxes]

        # Normalizing boxes
        mix_boxes_pad[:, 4] = (
            (mix_boxes_pad[:, 3] - mix_boxes_pad[:, 1])
            * (mix_boxes_pad[:, 2] - mix_boxes_pad[:, 0])
            / (image_w * image_h)
        )
        mix_boxes_pad[:, 0] = mix_boxes_pad[:, 0] / image_w
        mix_boxes_pad[:, 1] = mix_boxes_pad[:, 1] / image_h
        mix_boxes_pad[:, 2] = mix_boxes_pad[:, 2] / image_w
        mix_boxes_pad[:, 3] = mix_boxes_pad[:, 3] / image_h

        features = torch.tensor(mix_features_pad).float()
        image_masks = torch.tensor(image_mask).long()
        boxes = torch.tensor(mix_boxes_pad).float()

        return features, mix_num_boxes, boxes, image_masks
