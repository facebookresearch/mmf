# Pythia

<img src="info/pythia.jpg" width="40%">

The Pythia was the name of the high priestess of the Temple of Apollo at Delphi who also served as the oracle, commonly known as the Oracle of Delphi.

This repository contains the code, models and other data (features, annotations) required to reproduce the winning entry to the 2018 VQA Challenge (http://visualqa.org/roe.html) from the FAIR A-STAR team.
Our eventual goal is to build a software suite for VQA and Visual Dialog supporting many datasets and model architectures to enable easy, fair comparisons among them and accelerate research in this space.


## Getting Started

These instructions will get you a copy of the project and start running the models.


### Installing

1. Install Anaconda (Anaconda recommended: https://www.continuum.io/downloads).
2. Install cudnn/v7.0-cuda.9.0
3. Create envioment for vqa-suite
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

Download data, this step may take some time. (check the size of files at the end of readme)
```bash

git clone git@github.com:fairinternal/Pythia.git
cd Pythia

mkdir data
cd data
wget https://s3-us-west-1.amazonaws.com/vqa-suite/data/vqa2.0_glove.6B.300d.txt.npy
wget https://s3-us-west-1.amazonaws.com/vqa-suite/data/vocabulary_vqa.txt
wget https://s3-us-west-1.amazonaws.com/vqa-suite/data/answers_vqa.txt
wget https://s3-us-west-1.amazonaws.com/vqa-suite/data/imdb.tar.gz
wget https://s3-us-west-1.amazonaws.com/vqa-suite/data/rcnn_10_100.tar.gz
wget https://s3-us-west-1.amazonaws.com/vqa-suite/data/detectron.tar.gz
wget https://s3-us-west-1.amazonaws.com/vqa-suite/data/large_vocabulary_vqa.txt
wget https://s3-us-west-1.amazonaws.com/vqa-suite/data/large_vqa2.0_glove.6B.300d.txt.npy
gunzip imdb.tar.gz 
tar -xf imdb.tar

gunzip rcnn_10_100.tar.gz 
tar -xf rcnn_10_100.tar
rm -f rcnn_10_100.tar

gunzip detectron.tar.gz
tar -xf detectron.tar
rm -f detectron.tar
```

Get the help 
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
The results folder contains following info
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
The log files for tensorbord is stored under boards/



### Preprocess dataset
If you want to start from the original VQA dataset and preprocess data by yourself, use the follow instructions. 
***This part is not necessary if you download all data from quick start.***

#### VQA v2.0
 
Dowload dataset 
```bash
cd ../
mkdir -p orig_data/vqa_v2.0
cd orig_data/vqa_v2.0
./../../data_prep/vqa_v2.0/download_vqa_2.0.sh

```

preprocess dataset
```bash
cd ../../VQA_suite 
mkdir data

export PYTHONPATH=.

python data_prep/vqa_v2.0/extract_vocabulary.py \
--input_files ../orig_data/vqa_v2.0/v2_OpenEnded_mscoco_train2014_questions.json \
 ../orig_data/vqa_v2.0/v2_OpenEnded_mscoco_val2014_questions.json \
 ../orig_data/vqa_v2.0/v2_OpenEnded_mscoco_test2015_questions.json \
--out_dir data/

python data_prep/vqa_v2.0/process_answers.py \
--annotation_file ../orig_data/vqa_v2.0/v2_mscoco_train2014_annotations.json \
--val_annotation_file ../orig_data/vqa_v2.0/v2_mscoco_val2014_annotations.json  \
--out_dir data/ --min_freq 9

python data_prep/vqa_v2.0/extract_word_glove_embedding.py  \
--vocabulary_file data/vocabulary_vqa.txt  \
--glove_file ../orig_data/vqa_v2.0/glove/glove.6B.300d.txt \
--out_dir data/

python data_prep/vqa_v2.0/build_vqa_2.0_imdb.py --data_dir ../orig_data/vqa_v2.0/ --out_dir data/

```

Dowload image features
```bash
cd data/
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/rcnn_10_100.tar.gz
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/detectron_23.tar.gz
gunzip rcnn_10_100.tar.gz 
tar -xvf rcnn_10_100.tar
rm -f rcnn_10_100.tar

gunzip detectron.tar.gz
tar -xvf detectron.tar
rm -f detectron.tar


``` 


Training

Example:
```
python train.py 
```

### Test with pretrained models
| Description | performance (test-dev) | Link |
| --- | --- | --- |
|detectron_100_resnet_VG | 69.54 |https://s3-us-west-1.amazonaws.com/pythia-vqa/pretrained_models/detectron_100_resnet_VG.tar.gz
| baseline | 68.05 | https://s3-us-west-1.amazonaws.com/pythia-vqa/pretrained_models/baseline.tar.gz |
| baseline +VG +VD +mirror | 68.98 |https://s3-us-west-1.amazonaws.com/pythia-vqa/pretrained_models/most_data.tar.gz |
| detectron_finetune | 68.49 | https://s3-us-west-1.amazonaws.com/pythia-vqa/pretrained_models/detectron.tar.gz|
| detectron_finetune+VG |68.77 | https://s3-us-west-1.amazonaws.com/pythia-vqa/pretrained_models/detectron_VG.tar.gz |


#### Best Pretrained Model
The best pretrained Model can be download from:

```bash
mkdir pretrained_models/
cd pretrained_models
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/pretrained_models/most_data_models.tar.gz
gunzip most_data_models.tar.gz 
tar -xf most_data_models.tar
rm -f most_data_models.tar
```

get the resnet152 features for the best model and detectron features with fixed 100 bounding boxes
```bash
cd data
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/detectron_fix_100.tar.gz
gunzip detectron_fix_100.tar.gz
tar -xf detectron_fix_100.tar
rm -f detectron_fix_100.tar

wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/resnet152.tar.gz
gunzip resnet152.tar.gz
tar -xf resnet152.tar
rm -f resnet152.tar
```

```bash
mkdir pretrained_models/
cd pretrained_models
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/pretrained_models/detectron_100_resnet_VG.tar.gz
gunzip detectron_100_resnet_VG.tar.gz 
tar -xf detectron_100_resnet_VG.tar
rm -f detectron_100_resnet_VG.tar
```

test the best model with vqa test2015 datasset
```bash

python run_test.py --config pretrained_models/detectron_100_resnet_VG/567/config.yaml \
--model_path pretrained_models/detectron_100_resnet_VG/567/best_model.pth \
--out_prefix test_best_model
```

The result can be found as a json file test_best_model.json , 
this file can be uploaded to test on evalAI(https://evalai.cloudcv.org/web/challenges/challenge-page/80/submission)

### Ensemble different models
download all the models above
```bash
python ensemble.py --res_dirs pretrained_models/ --out ensemble_5.json
```
The result is in ensemble_5.json, check on evalAI, get the overall accuracy is 71.3

### Ensemble 30 models
To get all 30 pretrained models and the corresponding image features.The ensemble result for these 30 models is 72.18 on test-dev the models can be download:

```bash
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/ensembled.tar.gz
```
### Customize config
TO change models or adjust hyper-parameters, see [config_help.md](config_help.md)


### AWS s3 dataset summary
Here, we listed the size of some large files in our AWS S3 bucket.

| Description | size  |
| --- | --- | 
|data/rcnn_10_100.tar.gz | 71.0GB |
|data/detectron.tar.gz | 64.7GB|
|data/detectron_fix_100.tar.gz|98.1GB|
|data/resnet152.tar.gz | 399.6GB|
|ensembled.tar.gz| 462.1GB|

### Examples
![Alt text](info/vqa_example.png?raw=true "vqa examples")



