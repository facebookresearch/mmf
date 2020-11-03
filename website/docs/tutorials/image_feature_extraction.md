---
id: image_feature_extraction
title: Image Feature Extraction
sidebar_label: Image Feature Extraction
---

In this tutorial, we will go through the step-by-step process of generating image features with FasterRCNN feature extractors. MMF provides utility scripts for running image feature extractions using different models (Faster RCNN with X-101 backbones and X-152 backbones). For example: `tools/scripts/features/extract_features_vmb.py`. The script allows you to optionally parallelize the feature extraction execution.

Here are the steps in a nut shell:
- Prerequisites: setup
- Install vqa-maskrcnn-benchmark
- Download the dataset (not covered in this tutorial)
- Identify which vision feature extractor you'd like to use
- Extract image features
- Extract image features with slurm

## Prerequisites: setup

Create a new conda environment for feature extraction repo installation:

```bash
conda create -n maskrcnn_benchmark
conda activate maskrcnn_benchmark
```

A new conda environment is created so that the installation does not mess with the mmf conda environment.

Follow [this](https://www.internalfb.com/intern/staticdocs/mmf/docs/getting_started/installation#install-from-source-recommended) to install mmf in this new conda environment: maskrcnn_benchmark

## Install vqa-maskrcnn-benchmark

The following instructions is to install the maskrcnn-benchmark repo from [here](https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark).

### Requirements
- PyTorch >1.0 from a nightly release. Installation instructions can be found in [this](https://pytorch.org/get-started/locally/)
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- (optional) OpenCV for the webcam demo

### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib

# follow PyTorch installation in https://pytorch.org/get-started/locally/
conda install pytorch-nightly -c pytorch

# install torchvision
cd ~/github
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py install

# install pycocotools
cd ~/github
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd ~/github
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```

## Feature Extractors

We provide the model weights of two feature extractors based on [FasterRCNN](https://arxiv.org/pdf/1506.01497.pdf): Resnet101 and Resnet152. They are pretrained on [VisualGenome](https://arxiv.org/abs/1602.07332). In this tutorial, we use the FasterRCNN-ResNet101 feature extractor.

:::tip

To use a different feature extractor, you can override the `model_file` param to point at the feature extractor model file.

:::

## Extract Image Features

Images in `<FOLDER_PATH_TO_DATASET>` that have PNG/JPG/JPEG extensions will have their features extracted with the following invocation.

```bash
python mmf/mmf/tools/scripts/features/extract_features_vmb.py --model_name=X-152 --image_dir=<FOLDER_PATH_TO_DATASET> --output_folder=<OUTPUT_FOLDER>
```

## Extract Image Features with cluster workload manager (e.g., Slurm)

We can utilize slurms based cluster workload manager to do image feature extraction in parallel on multiple machines. This can greatly speed up the processing time if you have lots of images that need to have their features extracted. Please refer to `mmf/mmf/tools/scripts/features/extract_features_vmb.py` to see how you can adapt it to work for your purpose. As an example here, I showcase how to run image feature extraction on Flickr test set on 2 machines.

### Separate the images into 2 set

Create a `<IMAGE_LISTS_FOLDER>` folder that contains 2 files, each of the file contains a list of full image paths with newline as delimiter.

```bash
#!/bin/bash

for image_list in $(ls <IMAGE_LISTS_FOLDER>)
do
    sbatch --mem 128GB --nodes=1 --gres=gpu:1 --partition=<your_partition> --time=3000 --cpus-per-task=8 \
    flickr_test_extract_image_feature.sh $image_list
done
```

Separately, in `flickr_test_extract_image_feature.sh` write the following:

```bash

#!/bin/bash
python tools/scripts/features/extract_features_vmb.py \
    --image_dir $1 \
    --output_folder /checkpoint/ronghanghu/misc/open_images_vmb_feat/train \
    --model_file data/vmb_feat_extraction/detectron_model.pth \
    --config_file data/vmb_feat_extraction/detectron_model.yaml
```
