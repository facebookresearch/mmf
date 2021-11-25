---
id: image_feature_extraction_vinvl
title: Image Feature Extraction VinVL
sidebar_label: Image Feature Extraction VinVL
---

In this tutorial, we will go through the step-by-step process of generating image features with VinVL scene graph benchmark feature extractor.

Here are the steps in a nut shell:
- Prerequisites: setup
- Install scene-graph-benchmark (contains a modified maskrcnn_benchmark)
- Download your dataset
- Extract image features
- Extract image features with slurm

## Prerequisites: setup

Create a new conda environment for feature extraction repo installation:

```bash
conda create -n sg_benchmark
conda activate sg_benchmark
```

A new conda environment is created so that the installation does not mess with the mmf conda environment.

Install MMF in this environment.
Follow [this](https://mmf.sh/docs/#install-from-source-recommended)


## Install scene-graph-benchmark

The following instructions is to install the scene-graph-benchmark repo from [here](https://github.com/microsoft/scene_graph_benchmark/blob/main/INSTALL.md).

As a shortcut because you will have install pytorch and certain dependencies already through MMF, you maybe be able to follow:
```bash
pip install ninja yacs>=0.1.8 cython matplotlib tqdm opencv-python numpy>=1.19.5 timm einops pycocotools cityscapesscripts
git clone https://github.com/microsoft/scene_graph_benchmark
cd scene_graph_benchmark
python setup.py build develop
```

:::tip

The setup requires a lower version of gcc than 8, I used gcc/5.3.0.
You will also have to change `torch._six.PY3` => `torch._six.PY37` in `/your/path/scene_graph_benchmark/maskrcnn_benchmark/utils/imports.py`

:::


## Feature Extractors

We provide the model weights for VinVL's Scene-Graph-Benchmark AttrRCNN (based on ResNeXt-152 C4) model. It is pretrained on [COCO](https://arxiv.org/abs/1405.0312), [OpenImages](https://arxiv.org/abs/1811.00982), [Objects365](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.pdf), and [VisualGenome](https://arxiv.org/abs/1602.07332).

:::tip

To use a different feature extractor, you can override the `model_file` param to point at the feature extractor model file.

:::

## Extract Image Features

Images in `<FOLDER_PATH_TO_DATASET>` that have PNG/JPG/JPEG extensions will have their features extracted with the following invocation.

```bash
python mmf/mmf/tools/scripts/features/extract_features_vinvl.py --model_name=X-152-C4 --image_dir=<FOLDER_PATH_TO_DATASET> --output_folder=<OUTPUT_FOLDER>
```

## Extract Image Features with cluster workload manager (e.g., Slurm)

We can utilize slurms based cluster workload manager to do image feature extraction in parallel on multiple machines. This can greatly speed up the processing time if you have lots of images that need to have their features extracted. Please refer to `mmf/mmf/tools/scripts/features/extract_features_vinvl.py` to see how you can adapt it to work for your purpose. As an example here, I showcase how to run image feature extraction on Flickr test set on 2 machines.

The idea is that we partition our images into seperate sets, each used in a job submitted to the cluster.

### Separate the images into 2 sets

Create a `<IMAGE_LISTS_FOLDER>` folder that contains 2 files, each of the files containing a list of full image paths with newline as delimiter.

This is the bash script you will run to submit the jobs to your cluster.
If you name this script submit_extraction_jobs.sh, remember to `chmod +x ./submit_extraction_jobs.sh`.

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
python tools/scripts/features/extract_features_vinvl.py \
    --image_dir $1 \
    --output_folder /path/to/your/output/dir/
```

## Using the Extracted Features Directly

To use the extracted vinvl features (.npy files in your output dir) in a notebook or script instead of as part of an MMF dataset you can do the following.
The features and image information is stored in seperate .npy files, one for each image in your input list.
To load the extracted features for image `ex_0.jpg`,
```python
import numpy as np

feature_output_path = "/path/to/your/outputs/"
features = np.load(feature_output_path + 'ex_0.npy')
print(features.shape)
```

To load everything else extracted from the image, bounding boxs, labels, width, height, etc you can do the following.
```python
import numpy as np

feature_output_path = "/path/to/your/outputs/"
image_info = np.load(feature_output_path + 'ex_0_info.npy', allow_pickle=True)[()]
print(image_info.keys())
```
