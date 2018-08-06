#!/usr/bin/env bash

conda create --name vqa python=3.6
source activate vqa
pip install demjson pyyaml

pip install http://download.pytorch.org/whl/cu90/torch-0.3.0-cp36-cp36m-linux_x86_64.whl
pip install torchvision
pip install tensorboardX


