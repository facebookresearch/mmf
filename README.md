# Pythia

Pythia is a modular framework for Visual
Question Answering research, which formed the basis for
the winning entry to the VQA Challenge 2018 from Facebook AI Research (FAIR)’s A-STAR team. Please check our [paper](https://arxiv.org/abs/1807.09956) for more details.

(A-STAR: Agents that See, Talk, Act, and Reason.)

![Alt text](info/vqa_example.png?raw=true "vqa examples")


### Table of Contents
0. [Motivation](#motivation)
0. [Citing pythia](#citing-pythia)
0. [Installing pythia environment](#installing-pythia-environment)
0. [Quick start](#quick-start)
0. [Preprocess dataset](#preprocess-dataset)
0. [Test with pretrained models](#test-with-pretrained-models)
0. [Ensemble models](#ensemble-models)
0. [Customize config](#customize-config)
0. [Docker demo](#docker-demo)
0. [AWS s3 dataset summary](#aws-s3-dataset-summary)
0. [Acknowledgements](#acknowledgements)
0. [References](#references)


### Motivation
The motivation for Pythia comes from the following observation – a majority of today’s Visual Question Answering (VQA) models fit a particular design paradigm, with modules for question encoding, image feature extraction,
fusion of the two (typically with attention), and classification over the space of answers. The long-term goal of Pythia is to serve as a platform for easy and modular research & development in VQA and related directions like visual dialog.

#### Why the name _Pythia_?
The name Pythia is an homage to the Oracle
of Apollo at Delphi, who answered questions in Ancient
Greece. See [here](https://en.wikipedia.org/wiki/Pythia) for more details.

### Citing pythia
If you use Pythia in your research, please use the following BibTeX entries for reference:

The software:

```
@misc{pythia18software,
  title =        {Pythia},
  author =       {Yu Jiang and Vivek Natarajan and Xinlei Chen and Marcus Rohrbach and Dhruv Batra and Devi Parikh},
  howpublished = {\url{https://github.com/facebookresearch/pythia}},
  year =         {2018}
}
```

The technical report detailing the description and analysis for our winning entry to the VQA 2018 challenge:

```
@article{pythia18arxiv,
  title =        {Pythia v0.1: the Winning Entry to the VQA Challenge 2018},
  author =       {{Yu Jiang*} and {Vivek Natarajan*} and {Xinlei Chen*} and Marcus Rohrbach and Dhruv Batra and Devi Parikh},
  journal =      {arXiv preprint arXiv:1807.09956},
  year =         {2018}
}
```

\* Yu Jiang, Vivek Natarajan and Xinlei Chen contributed equally to the winning entry to the VQA 2018 challenge.

### Installing pythia environment

1. Install Anaconda (Anaconda recommended: https://www.continuum.io/downloads).
2. Install cudnn v7.0 and cuda.9.0
3. Create environment for pythia
```bash
conda create --name vqa python=3.6

source activate vqa
pip install demjson pyyaml

pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl

pip install torchvision
pip install tensorboardX

```


### Quick start
We provide preprocessed data files to directly start training and evaluating. Instead of using the original `train2014` and `val2014` splits, we split `val2014` into `val2train2014` and `minival2014`, and use `train2014` + `val2train2014` for training and `minival2014` for validation.

Download data. This step may take some time. Check the sizes of files at the end of readme.
```bash

git clone git@github.com:facebookresearch/pythia.git
cd Pythia

mkdir data
cd data
wget https://dl.fbaipublicfiles.com/pythia/data/vqa2.0_glove.6B.300d.txt.npy
wget https://dl.fbaipublicfiles.com/pythia/data/vocabulary_vqa.txt
wget https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt
wget https://dl.fbaipublicfiles.com/pythia/data/imdb.tar.gz
wget https://dl.fbaipublicfiles.com/pythia/features/rcnn_10_100.tar.gz
wget https://dl.fbaipublicfiles.com/pythia/features/detectron_fix_100.tar.gz
wget https://dl.fbaipublicfiles.com/pythia/data/large_vocabulary_vqa.txt
wget https://dl.fbaipublicfiles.com/pythia/data/large_vqa2.0_glove.6B.300d.txt.npy
gunzip imdb.tar.gz 
tar -xf imdb.tar

gunzip rcnn_10_100.tar.gz 
tar -xf rcnn_10_100.tar
rm -f rcnn_10_100.tar

gunzip detectron.tar.gz
tar -xf detectron.tar
rm -f detectron.tar
```

Optional command-line arguments for `train.py`
```bash
python train.py -h

usage: train.py [-h] [--config CONFIG] [--out_dir OUT_DIR] [--seed SEED]
                [--config_overwrite CONFIG_OVERWRITE] [--force_restart]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config yaml file
  --out_dir OUT_DIR     output directory, default is current directory
  --seed SEED           random seed, default 1234, set seed to -1 if need a
                        random seed between 1 and 100000
  --config_overwrite CONFIG_OVERWRITE
                        a json string to update yaml config file
  --force_restart       flag to force clean previous result and restart
                        training
```

Run model without finetuning
```bash
cd ../
python train.py
```
If there is a out of memory error, try:
```bash
python train.py --config_overwrite '{data:{image_fast_reader:false}}'
```

Run model with features from detectron with finetuning
```bash
python train.py --config config/keep/detectron.yaml

```
Check result for the default run
```bash
cd results/default/1234

```
The results folder contains the following info
```angular2html
results
|_ default
|  |_ 1234 (default seed)
|  |  |_config.yaml
|  |  |_best_model.pth
|  |  |_best_model_predict_test.pkl 
|  |  |_best_model_predict_test.json (json file for predicted results on test dataset)
|  |  |_model_00001000.pth (mpdel snapshot at iter 1000)
|  |  |_result_on_val.txt
|  |  |_ ...
|  |_(other_cofig_setting)
|  |  |_...
|_ (other_config_file)
|

```
The log files for tensorbord are stored under `boards/`


### Preprocess dataset
If you want to start from the original VQA dataset and preprocess data by yourself, use the following instructions in [data_preprocess.md](data_prep/data_preprocess.md). 
***This part is not necessary if you download all data from quick start.***


### Test with pretrained models
Note: all of these models below are trained with validation set included

| Description | performance (test-dev) | Link |
| --- | --- | --- |
|detectron_100_resnet_most_data | 70.01 |https://dl.fbaipublicfiles.com/pythia/pretrained_models/detectron_100_resnet_most_data.tar.gz
| baseline | 68.05 | https://dl.fbaipublicfiles.com/pythia/pretrained_models/baseline.tar.gz |
| baseline +VG +VisDal +mirror | 68.98 |https://dl.fbaipublicfiles.com/pythia/pretrained_models/most_data.tar.gz |
| detectron_finetune | 68.49 | https://dl.fbaipublicfiles.com/pythia/pretrained_models/detectron.tar.gz|
| detectron_finetune+VG +VisDal +mirror |69.24 | https://dl.fbaipublicfiles.com/pythia/pretrained_models/detectron_most_data.tar.gz |


#### Best Pretrained Model
The best pretrained model can be downloaded as follows:

```bash
mkdir pretrained_models/
cd pretrained_models
wget https://dl.fbaipublicfiles.com/pythia/pretrained_models/detectron_100_resnet_most_data.tar.gz
gunzip detectron_100_resnet_most_data.tar.gz 
tar -xf detectron_100_resnet_most_data.tar
rm -f detectron_100_resnet_most_data.tar
```


Get ResNet152 features and Detectron features with fixed 100 bounding boxes
```bash
cd data
wget https://dl.fbaipublicfiles.com/pythia/features/detectron_fix_100.tar.gz
gunzip detectron_fix_100.tar.gz
tar -xf detectron_fix_100.tar
rm -f detectron_fix_100.tar

wget https://dl.fbaipublicfiles.com/pythia/features/resnet152.tar.gz
gunzip resnet152.tar.gz
tar -xf resnet152.tar
rm -f resnet152.tar
```


Test the best model on the VQA test2015 dataset
```bash

python run_test.py --config pretrained_models/detectron_100_resnet_most_data/1234/config.yaml \
--model_path pretrained_models/detectron_100_resnet_most_data/1234/best_model.pth \
--out_prefix test_best_model
```

The results will be saved as a json file `test_best_model.json`, and this file can be uploaded to the evaluation server on EvalAI (https://evalai.cloudcv.org/web/challenges/challenge-page/80/submission).

### Ensemble models
Download all the models above
```bash
python ensemble.py --res_dirs pretrained_models/ --out ensemble_5.json
```
Results will be saved in `ensemble_5.json`. This ensemble can get accuracy 71.65 on test-dev.

### Customize config
To change models or adjust hyper-parameters, see [config_help.md](config_help.md)

### Docker demo
To quickly tryout a model interactively with [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

```bash
git clone https://github.com/facebookresearch/pythia.git
nvidia-docker build pythia -t pythia:latest
nvidia-docker run -ti --net=host pythia:latest
```

This will open a jupyter notebook with a demo model to which you can ask questions interactively.

### AWS s3 dataset summary
Here, we listed the size of some large files in our AWS S3 bucket.

| Description | size  |
| --- | --- | 
|data/rcnn_10_100.tar.gz | 71.0GB |
|data/detectron.tar.gz | 106.2 GB|
|data/detectron_fix_100.tar.gz|162.6GB|
|data/resnet152.tar.gz | 399.6GB|

### Acknowledgements
We would like to thank Peter Anderson, Abhishek Das, Stefan Lee, Jiasen Lu, Jianwei Yang, Licheng Yu, 
Luowei Zhou for helpful discussions, Peter Anderson for providing
training data for the Visual Genome detector, Deshraj Yadav
for responses on EvalAI related questions, Stefan Lee
for suggesting the name *Pythia*, Meet Shah for building the docker demo for Pythia and 
Abhishek Das, Abhishek Kadian for feedback on our codebase.


### References
- Y. Jiang, and V. Natarajan and X. Chen and M. Rohrbach and D. Batra and D. Parikh. Pythia v0.1: The Winning Entry to the VQA Challenge 2018. CoRR, abs/1807.09956, 2018.
- P. Anderson, X. He, C. Buehler, D. Teney, M. Johnson, S. Gould, and L. Zhang. Bottom-up and top-down attenttion for image captioning and visual question answering. In _CVPR_, 2018.
- S. Antol,   A. Agrawal,   J. Lu,   M. Mitchell,   D. Batra,C. Lawrence Zitnick, and D. Parikh.  VQA:  Visual question answering. In _ICCV_, 2015
- A. Das,  S. Kottur,  K. Gupta,  A. Singh,  D. Yadav,  J. M. Moura, D. Parikh, and D. Batra.  Visual Dialog.  In _CVPR_, 2017
- Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In _CVPR_, 2017.
- D. Teney, P. Anderson, X. He, and A. van den Hengel. Tips and tricks for visual question answering: Learnings from the 2017 challenge. CoRR, abs/1708.02711, 2017.
- Z. Yu, J. Yu, C. Xiang, J. Fan, and D. Tao. Beyond bilinear: Generalized multimodal factorized high-order pooling for visual question answering. _TNNLS_, 2018.



