# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

import skimage.color
import skimage.io
from global_variables.global_variables import use_cuda

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)

args = parser.parse_args()
gpu_id = args.gpu_id  # set GPU id to use
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
sys.path.append("../../")

image_basedir = args.data_dir
save_basedir = args.out_dir

channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)


class vgg16_feature_module(nn.Module):
    def __init__(self, vgg16_model):
        super(vgg16_feature_module, self).__init__()
        self.feature_module = nn.Sequential(*list(list(vgg16_model.children())[0]))

    def forward(self, x):
        return self.feature_module(x)


vgg16 = models.vgg16(pretrained=True)
vgg16_feature = vgg16_feature_module(vgg16)
vgg16_feature = vgg16_feature.cuda() if use_cuda else vgg16_feature


def extract_image_pool5(impath):
    im = skimage.io.imread(impath)[..., :3]
    im_val = im[np.newaxis, ...] - channel_mean

    # permute to get NCHW
    im_val = np.transpose(im_val, axes=(0, 3, 1, 2))
    im_val_tensor = torch.FloatTensor(im_val)
    im_val_variable = Variable(im_val_tensor)
    im_val_variable = im_val_variable.cuda() if use_cuda else im_val_variable

    pool5_val = vgg16_feature(im_val_variable)
    return pool5_val.data.cpu().numpy()


def extract_dataset_pool5(image_dir, save_dir, ext_filter="*.png"):
    image_list = glob(image_dir + "/" + ext_filter)
    os.makedirs(save_dir, exist_ok=True)

    for n_im, impath in enumerate(image_list):
        if (n_im + 1) % 100 == 0:
            print("processing %d / %d" % (n_im + 1, len(image_list)))
        image_name = os.path.basename(impath).split(".")[0]
        save_path = os.path.join(save_dir, image_name + ".npy")
        if not os.path.exists(save_path):
            pool5_val = extract_image_pool5(impath)
            np.save(save_path, pool5_val)


for image_set in ["train", "val", "test"]:
    print("Extracting image set " + image_set)
    extract_dataset_pool5(
        os.path.join(image_basedir, image_set), os.path.join(save_basedir, image_set)
    )
    print("Done.")
