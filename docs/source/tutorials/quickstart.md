# Quickstart [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Z9fsh10rFtgWe4uy8nvU4mQmqdokdIRR) [![](https://circleci.com/gh/facebookresearch/pythia.svg?style=svg)](https://circleci.com/gh/facebookresearch/pythia)


**Authors**: Amanpreet Singh

In this quickstart, we are going to train LoRRA model on TextVQA. Follow instructions at the bottom
to train other models in Pythia.

## Demo

1. [Pythia VQA](https://colab.research.google.com/drive/1Z9fsh10rFtgWe4uy8nvU4mQmqdokdIRR) 
2. [BUTD Captioning](https://colab.research.google.com/drive/1vzrxDYB0vxtuUy8KCaGxm--nDCJvyBSg)

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

```eval_rst
.. note::

  1. If you face any issues with the setup, check the Troubleshooting/FAQ section below.
  2. You can also create/activate your own conda environments before running
     above commands.
```

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

## Troubleshooting/FAQs

1. If `setup.py` causes any issues, please install fastText first directly from the source and
then run `python setup.py develop`. To install fastText run following commands:

```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install -e .
```

## Next steps

To dive deep into world of Pythia, you can move on the following next topics:

- [Concepts and Terminology](./concepts)
- [Using Pretrained Models](./pretrained_models)
- [Challenge Participation](./challenge)

## Citation

If you use Pythia in your work, please cite:

```text
@inproceedings{Singh2019TowardsVM,
  title={Towards VQA Models That Can Read},
  author={Singh, Amanpreet and Natarajan, Vivek and Shah, Meet and Jiang, Yu and Chen, Xinlei and Batra, Dhruv and Parikh, Devi and Rohrbach, Marcus},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

and

```text
@inproceedings{singh2019pythia,
  title={Pythia-a platform for vision \& language research},
  author={Singh, Amanpreet and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  booktitle={SysML Workshop, NeurIPS},
  volume={2018},
  year={2019}
}
```
