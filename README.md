# Pythia

Pythia is a modular framework for vision and language multimodal research. Built on top
of PyTorch, pythia features:

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

## Getting Started

First install the repo using

```
git clone https://github.com/facebookresearch/pythia

# You can also create your own conda environment and then enter this step
cd pythia && pip install -r requirements.txt
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
# Check if we are in the correct folder
# This should show pythia root folder
pwd
mkdir -p data && cd data
wget http://dl.fbaipublicfiles.com/pythia/data/vocab.tar.gz

# Should result in vocabs folder in your data dir
tar xf vocab.tar.gz

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
| Pythia | pythia    | vqa2, vizwiz, textvqa | Coming soon!      |                                                           |
| LoRRA  | lorra     | vizwiz, textvqa       | Coming soon!      | vqa2 support is coming soon!                              |
| BAN    | ban       | vqa2, vizwiz, textvqa | Coming soon!      | Support is preliminary and haven't been tested throughly. |

For running `LoRRA` on `TextVQA`, run the following command from root directory of your pythia clone:

```
pythia tools/run.py --tasks vqa --datasets vqa2 --model lorra --config configs/vqa/textvqa/lorra.yml
```

## Documentation

Documentation specific on how to navigate around Pythia and making changes will be available soon.

## Citation

If you use Pythia in your work, please cite:

```
@inproceedings{Singh2019TowardsVM,
  title={Towards VQA Models that can Read},
  author={Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, Marcus Rohrbach},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

and

```
@inproceedings{singh2019pythia,
  title={Pythia-a platform for vision \& language research},
  author={Singh, Amanpreet and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  booktitle={SysML Workshop, NeurIPS},
  volume={2018},
  year={2019}
}
```

## Miscellaneous
If you are looking for v0.1, the VQA 2018 challenge winner entry, checkout the tag
for [v0.1](https://github.com/facebookresearch/pythia/releases/tag/v0.1).

## License

Pythia is licensed under BSD license available in [LICENSE](LICENSE) file
