# Copyright (c) Facebook, Inc. and its affiliates.

"""
 coding=utf-8
 Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal
 Adapted From Facebook Inc, Detectron2 && Huggingface Co.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.import copy
 """

from typing import List

import omegaconf
import torch
from torch import nn
from torch.nn import functional as F


try:
    from detectron2.layers.batch_norm import get_norm
    from detectron2.layers.wrappers import Conv2d
    from detectron2.modeling import ShapeSpec
    from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
    from detectron2.modeling.proposal_generator.rpn import RPN, StandardRPNHead
    from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads
    from detectron2.structures.image_list import ImageList
except ImportError:
    pass


def build_backbone(config: omegaconf.DictConfig):
    """
    Difference between this and the build_backbone provided
    by detectron2 is as follows:
    - Different stem, include caffe_maxpool
    - Number of blocks is different, unconfigurable in detectron
    - Freeze-at operates differently in detectron
    """
    input_shape = ShapeSpec(channels=len(config.MODEL.PIXEL_MEAN))
    norm = config.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=config.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
        caffe_maxpool=config.MODEL.MAX_POOL,
    )
    freeze_at = config.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False

    out_features = config.MODEL.RESNETS.OUT_FEATURES
    depth = config.MODEL.RESNETS.DEPTH
    num_groups = config.MODEL.RESNETS.NUM_GROUPS
    width_per_group = config.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = config.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = config.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = config.MODEL.RESNETS.RES5_DILATION
    assert res5_dilation in {1, 2}, f"res5_dilation cannot be {res5_dilation}."

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[
        depth
    ]

    stages = []
    out_stage_idx = [
        {"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features
    ]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "num_groups": num_groups,
            "norm": norm,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation,
        }

        stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        stages.append(blocks)

    return ResNet(stem, stages, out_features=out_features, freeze_at=-1)


class BasicStem(nn.Module):
    """
    The differences between this and detectron:
    - The forward method uses caffe_maxpool
      this is not configurable in detectron
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        norm: str = "BN",
        caffe_maxpool: bool = False,
    ):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        self.caffe_maxpool = caffe_maxpool
        # use pad 1 instead of pad zero

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu_(x)
        if self.caffe_maxpool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True)
        else:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool


class GeneralizedRCNN(nn.Module):
    def __init__(self, config: omegaconf.DictConfig):
        super().__init__()
        self.device = torch.device(config.MODEL.DEVICE)
        self.backbone = build_backbone(config)
        self.proposal_generator = RPN(config, self.backbone.output_shape())
        self._fix_proposal_generator(config)
        self.roi_heads = Res5ROIHeads(config, self.backbone.output_shape())
        self._fix_res5_block(config)
        self.to(self.device)

    def _fix_proposal_generator(self, config: omegaconf.DictConfig):
        in_channels = [
            val.channels for key, val in self.backbone.output_shape().items()
        ]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]
        if config.MODEL.PROPOSAL_GENERATOR.HIDDEN_CHANNELS == -1:
            hid_channels = in_channels
        else:
            hid_channels = config.MODEL.PROPOSAL_GENERATOR.HIDDEN_CHANNELS
        self.proposal_generator.rpn_head.conv = nn.Conv2d(
            in_channels, hid_channels, kernel_size=3, stride=1, padding=1
        )
        shape = self.backbone.output_shape()
        features = config.MODEL.RPN.IN_FEATURES
        example_head = StandardRPNHead.from_config(config, [shape[f] for f in features])
        num_cell_anchors = example_head["num_anchors"]
        box_dim = example_head["box_dim"]
        self.proposal_generator.rpn_head.objectness_logits = nn.Conv2d(
            hid_channels, num_cell_anchors, kernel_size=1, stride=1
        )
        self.proposal_generator.rpn_head.anchor_deltas = nn.Conv2d(
            hid_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1
        )

    def _fix_res5_block(self, config: omegaconf.DictConfig):
        res5_halve = config.MODEL.ROI_BOX_HEAD.RES5HALVE
        if not res5_halve:
            """
            Modifications for VG in RoI heads:
            1. Change the stride of conv1 and shortcut in Res5.Block1 from 2 to 1
            2. Modifying all conv2 with (padding: 1 --> 2) and (dilation: 1 --> 2)
            """
            self.roi_heads.res5[0].conv1.stride = (1, 1)
            self.roi_heads.res5[0].shortcut.stride = (1, 1)
            for i in range(3):
                self.roi_heads.res5[i].conv2.padding = (2, 2)
                self.roi_heads.res5[i].conv2.dilation = (2, 2)

    def forward_for_roi_head(self, features: List, proposal_boxes: List):
        box_features = self.roi_heads._shared_roi_transform(features, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        return feature_pooled

    def forward(
        self,
        images: torch.Tensor,
        image_shapes: torch.Tensor,
        gt_boxes: torch.Tensor = None,
        proposals: torch.Tensor = None,
        scales_yx: torch.Tensor = None,
        **kwargs,
    ):
        """
        kwargs:
            max_detections (int), return_tensors {"np", "pt", None}, padding {None,
            "max_detections"}, pad_value (int), location = {"cuda", "cpu"}
        """
        if self.training:
            raise NotImplementedError()
        return self.inference(
            images=images,
            image_shapes=image_shapes,
            gt_boxes=gt_boxes,
            proposals=proposals,
            scales_yx=scales_yx,
            **kwargs,
        )

    @torch.no_grad()
    def inference(
        self,
        images: torch.Tensor,
        image_shapes: torch.Tensor,
        gt_boxes: torch.Tensor = None,
        proposals: torch.Tensor = None,
        scales_yx: torch.Tensor = None,
        **kwargs,
    ):
        # run images through backbone
        features = self.backbone(images)

        image_list = ImageList(images, image_shapes)

        # generate proposals if none are available
        if proposals is None:
            proposal_boxes, _ = self.proposal_generator(image_list, features, gt_boxes)
        else:
            assert proposals is not None

        proposal_boxes = [proposal_boxes[0].get_fields()["proposal_boxes"]]

        feature_pooled = self.forward_for_roi_head([features["res4"]], proposal_boxes)

        preds_per_image = [p.size(0) for p in [proposal_boxes[0].tensor]]

        roi_features = feature_pooled.split(preds_per_image, dim=0)

        return roi_features
