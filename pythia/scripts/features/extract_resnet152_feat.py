# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

TARGET_IMAGE_SIZE = [448, 448]
CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose(
    [
        transforms.Resize(TARGET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
    ]
)

use_cuda = torch.cuda.is_available()

# NOTE feat path "https://download.pytorch.org/models/resnet152-b121ed2d.pth"
RESNET152_MODEL = models.resnet152(pretrained=True)
RESNET152_MODEL.eval()

if use_cuda:
    RESNET152_MODEL = RESNET152_MODEL.cuda()


class ResNet152FeatModule(nn.Module):
    def __init__(self):
        super(ResNet152FeatModule, self).__init__()
        modules = list(RESNET152_MODEL.children())[:-2]
        self.feature_module = nn.Sequential(*modules)

    def forward(self, x):
        return self.feature_module(x)


_resnet_module = ResNet152FeatModule()
if use_cuda:
    _resnet_module = _resnet_module.cuda()


def extract_image_feat(img_file):
    img = Image.open(img_file).convert("RGB")
    img_transform = data_transforms(img)
    # make sure grey scale image is processed correctly
    if img_transform.shape[0] == 1:
        img_transform = img_transform.expand(3, -1, -1)
    img_var = Variable(img_transform.unsqueeze(0))
    if use_cuda:
        img_var = img_var.cuda()

    img_feat = _resnet_module(img_var)
    return img_feat


def get_image_id(image_name):
    image_id = int(image_name.split(".")[0].split("_")[-1])
    return image_id


def extract_dataset_pool5(image_dir, save_dir, total_group, group_id, ext_filter):
    image_list = glob(image_dir + "/*." + ext_filter)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for n_im, impath in enumerate(image_list):
        if (n_im + 1) % 100 == 0:
            print("processing %d / %d" % (n_im + 1, len(image_list)))
        image_name = os.path.basename(impath)
        image_id = get_image_id(image_name)
        if image_id % total_group != group_id:
            continue

        feat_name = image_name.replace(ext_filter, "npy")
        save_path = os.path.join(save_dir, feat_name)
        tmp_lock = save_path + ".lock"

        if os.path.exists(save_path) and not os.path.exists(tmp_lock):
            continue
        if not os.path.exists(tmp_lock):
            os.makedirs(tmp_lock)

        # pool5_val = extract_image_feat(impath).permute(0, 2, 3, 1)
        try:
            pool5_val = extract_image_feat(impath).permute(0, 2, 3, 1)
        except:
            print("error for" + image_name)
            continue

        feat = pool5_val.data.cpu().numpy()
        np.save(save_path, feat)
        os.rmdir(tmp_lock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_group", type=int, default=1)
    parser.add_argument("--group_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--image_ext", type=str, default="jpg")

    args = parser.parse_args()

    extract_dataset_pool5(
        args.data_dir, args.out_dir, args.total_group, args.group_id, args.image_ext
    )
