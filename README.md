# Pythia

[![Documentation Status](https://readthedocs.org/projects/learnpythia/badge/?version=latest)](https://learnpythia.readthedocs.io/en/latest/?badge=latest) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Z9fsh10rFtgWe4uy8nvU4mQmqdokdIRR)


Pythia is a modular framework for vision and language multimodal research. Built on top
of PyTorch, it features:

- **Model Zoo**: Reference implementations for state-of-the-art vision and language model including
[LoRRA](https://arxiv.org/abs/1904.08920) (SoTA on VQA and TextVQA),
[Pythia](https://arxiv.org/abs/1807.09956) model (VQA 2018 challenge winner) and [BAN]().
- **Multi-Tasking**: Support for multi-tasking which allows training on multiple dataset together.
- **Datasets**: Includes support for various datasets built-in including VQA, VizWiz, TextVQA and VisualDialog.
- **Modules**: Provides implementations for many commonly used layers in vision and language domain
- **Distributed**: Support for distributed training based on DataParallel as well as DistributedDataParallel.
- **Unopinionated**: Unopinionated about the dataset and model implementations built on top of it.
- **Customization**: Custom losses, metrics, scheduling, optimizers, tensorboard; suits all your custom needs.

You can use Pythia to **_bootstrap_** for your next vision and language multimodal research project.

Pythia can also act as **starter codebase** for challenges around vision and
language datasets (TextVQA challenge, VQA challenge)

## Documentation

Learn more about Pythia [here](https://learnpythia.readthedocs.io/en/latest/).

## Demo

Try the demo for Pythia model on [Colab](https://colab.research.google.com/drive/1Z9fsh10rFtgWe4uy8nvU4mQmqdokdIRR).

## Getting Started

First install the repo using

```
git clone https://github.com/facebookresearch/pythia ~/pythia

# You can also create your own conda environment and then enter this step
cd ~/pythia
python setup.py develop
```

Now, Pythia should be ready to use. Follow steps in specific sections to start training
your own models using Pythia.


## Data

Default configuration assume that all of the data is present in the `data` folder inside `pythia` folder.

Depending on which dataset you are planning to use download the feature and imdb (image database) data for that particular dataset using
the links in the table (_right click -> copy link address_). Feature data has been extracted out from Detectron and are used in the
reference models. Example below shows the sample commands to be run, once you have
the feature (feature_link) and imdb (imdb_link) data links.

```
cd ~/pythia
mkdir -p data && cd data
wget http://dl.fbaipublicfiles.com/pythia/data/vocab.tar.gz

# Should result in vocabs folder in your data dir
tar xf vocab.tar.gz

# Download detectron weights
wget http://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf detectron_weights.tar.gz

# Now download the features required, feature link is taken from the table below
# These two commands below can take time
wget feature_link

# [features].tar.gz is the file you just downloaded, replace that with your file's name
tar xf [features].tar.gz

# Make imdb folder and download required imdb
mkdir -p imdb && cd imdb
wget imdb_link

# [imdb].tar.gz is the file you just downloaded, replace that with your file's name
tar xf [imdb].tar.gz
```

| Dataset      | Key | Task | ImDB Link                                                                         | Features Link                                                                   |
|--------------|-----|-----|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| TextVQA      | textvqa | vqa | [TextVQA 0.5 ImDB](https://dl.fbaipublicfiles.com/pythia/data/imdb/textvqa_0.5.tar.gz) | [OpenImages](https://dl.fbaipublicfiles.com/pythia/features/open_images.tar.gz) |
| VQA 2.0      | vqa2 | vqa | [VQA 2.0 ImDB](https://dl.fbaipublicfiles.com/pythia/data/imdb/vqa.tar.gz)                 | [COCO](https://dl.fbaipublicfiles.com/pythia/features/coco.tar.gz)              |
| VizWiz       | vizwiz | vqa | [VizWiz ImDB](https://dl.fbaipublicfiles.com/pythia/data/imdb/vizwiz.tar.gz)           | [VizWiz](https://dl.fbaipublicfiles.com/pythia/features/vizwiz.tar.gz)          |
| VisualDialog | visdial | dialog | Coming soon!                                                                      | Coming soon!                                                                    |


## Training

Once we have the data downloaded and in place, we just need to select a model, an appropriate task and dataset as well related config file. Default configurations can be found  inside `configs` folder in repository's root folder. Configs are divided for models in format of `[task]/[dataset_key]/[model_key].yml` where `dataset_key` can be retrieved from the table above. For example, for `pythia` model, configuration for VQA 2.0 dataset can be found at `configs/vqa/vqa2/pythia.yml`. Following table shows the keys and the datasets
supported by the models in Pythia's model zoo.

| Model  | Key | Supported Datasets    | Pretrained Models | Notes                                                     |
|--------|-----------|-----------------------|-------------------|-----------------------------------------------------------|
| Pythia | pythia    | vqa2, vizwiz, textvqa | [vqa2 train+val](https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/pythia_train_val.pth), [vqa2 train only](https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia.pth)      |                                                            |
| LoRRA  | lorra     | vqa2, vizwiz, textvqa       | [textvqa](https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/lorra_best.pth)      |                               |
| BAN    | ban       | vqa2, vizwiz, textvqa | Coming soon!      | Support is preliminary and haven't been tested thoroughly. |


For running `LoRRA` on `TextVQA`, run the following command from root directory of your pythia clone:

```
cd ~/pythia
python tools/run.py --tasks vqa --datasets textvqa --model lorra --config configs/vqa/textvqa/lorra.yml
```

## Pretrained Models

We are including some of the pretrained models as described in the table above.
For e.g. to run the inference using LoRRA for TextVQA for EvalAI use following commands:

```
# Download the model first
cd ~/pythia/data
mkdir -p models && cd models;
# Get link from the table above and extract if needed
wget https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/lorra_best.pth

cd ../..
# Replace tasks, datasets and model with corresponding key for other pretrained models
python tools/run.py --tasks vqa --datasets textvqa --model lorra --config configs/vqa/textvqa/lorra.yml \
--run_type inference --evalai_inference 1 --resume_file data/models/lorra_best.pth
```


## Documentation

Documentation specific on how to navigate around Pythia and making changes will be available soon.

## Citation

If you use Pythia in your work, please cite:

```
@inproceedings{singh2019TowardsVM,
  title={Towards VQA Models That Can Read},
  author={Singh, Amanpreet and Natarajan, Vivek and Shah, Meet and Jiang, Yu and Chen, Xinlei and Batra, Dhruv and Parikh, Devi and Rohrbach, Marcus},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

and

```
@inproceedings{singh2018pythia,
  title={Pythia-a platform for vision \& language research},
  author={Singh, Amanpreet and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  booktitle={SysML Workshop, NeurIPS},
  volume={2018},
  year={2018}
}
```

## Miscellaneous
If you are looking for v0.1, the VQA 2018 challenge winner entry, checkout the tag
for [v0.1](https://github.com/facebookresearch/pythia/releases/tag/v0.1).

## Troubleshooting/FAQs

1. If `setup.py` causes any issues, please install fastText first directly from the source and
then run `python setup.py develop`. To install fastText run following commands:

```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install -e .
```

## License

Pythia is licensed under BSD license available in [LICENSE](LICENSE) file
