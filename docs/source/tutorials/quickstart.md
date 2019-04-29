# Quickstart

**Authors**: Amanpreet Singh

In this quickstart, we are going to train LoRRA model on TextVQA. Follow instructions at the bottom
to train other models in Pythia.

## Installation

1. Clone Pythia repository

```bash
git clone https://github.com/facebookresearch/pythia ~/pythia
```

2. Install dependencies and setup

```bash
cd ~/pythia
python setup.py develop
```

**Note:** You can also create/activate your own conda environments before running
above commands.

## Getting Data

Datasets currently supported in Pythia require two parts of data, features and ImDB.
Features correspond to pre-extracted object features from an object detector. ImDB
is the image database for the datasets which contains information such as questions
and answers in case of TextVQA.

For TextVQA, we need to download features for OpenImages' images which are included
in it and TextVQA 0.5 ImDB. We assume that all of the data is kept inside `data`
folder under `pythia` root folder. Table in bottom shows corresponding features
and ImDB links for datasets supported in pythia.

```bash
cd ~/pythia;
# Create data folder
mkdir -p data && cd data;

# Download and extract the features
wget https://dl.fbaipublicfiles.com/pythia/features/open_images.tar.gz
tar xf open_images.tar.gz

# Get vocabularies
wget http://dl.fbaipublicfiles.com/pythia/data/vocab.tar.gz
tar xf vocab.tar.gz

# Download detectron weights required by some models
wget http://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf detectron_weights.tar.gz

# Download and extract ImDB
mkdir -p imdb && cd imdb
wget https://dl.fbaipublicfiles.com/pythia/data/imdb/textvqa_0.5.tar.gz
tar xf textvqa_0.5.tar.gz
```

## Training

Once we have the data in-place, we can start training by running the following command:

```bash
cd ~/pythia;
python tools/run.py --tasks vqa --datasets textvqa --model lorra --config \
configs/vqa/textvqa/lorra.yml
```

## Inference

For running inference or generating predictions for EvalAI, we can download
a corresponding pretrained model and then run the following commands:

```bash
cd ~/pythia/data
mkdir -p models && cd models;
wget https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/lorra_best.pth
cd ../..
python tools/run.py --tasks vqa --datasets textvqa --model lorra --config \
configs/vqa/textvqa/lorra.yml --resume_file data/models/lorra_best.pth \
--evalai_inference 1 --run_type inference
```

For running inference on `val` set, use `--run_type val` and rest of the arguments remain same.
Check more details in [pretrained models](pretrained_models) section.

These commands should be enough to get you started with training and performing inference using Pythia.

## Next steps

To dive deep into world of Pythia, you can move on the following next topics:

- [Concepts and Terminology](./concepts)
- [Using Pretrained Models](./pretrained_models)
- [Challenge Participation](./challenge)
