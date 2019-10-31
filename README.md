<div align="center">
  <a href="https://readthedocs.org/projects/learnpythia/">
    <img width="60%" alt="Pythia" src="https://i.imgur.com/wPgp4N4.png"/>
  </a>
</div>

<div align="center">
  <a href="https://learnpythia.readthedocs.io/en/latest/?badge=latest">
  <img alt="Documentation Status" src="https://img.shields.io/readthedocs/pythia.svg?label=docs&style=flat&logo=read-the-docs"/>
  </a>
  <a href="https://colab.research.google.com/drive/1Z9fsh10rFtgWe4uy8nvU4mQmqdokdIRR">
  <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"/>
  </a>
  <a href="https://circleci.com/gh/facebookresearch/pythia">
  <img alt="CircleCI" src="https://circleci.com/gh/facebookresearch/pythia.svg?style=svg"/>
  </a>
</div>

Pythia is a modular framework for vision and language multimodal research. Built on top
of PyTorch, it features:

- **Model Zoo**: Reference implementations for state-of-the-art vision and language model including
[LoRRA](https://arxiv.org/abs/1904.08920) (SoTA on VQA and TextVQA),
[Pythia](https://arxiv.org/abs/1807.09956) model (VQA 2018 challenge winner) , [BAN](https://arxiv.org/abs/1805.07932) and [BUTD](https://arxiv.org/abs/1707.07998).
- **Multi-Tasking**: Support for multi-tasking which allows training on multiple dataset together.
- **Datasets**: Includes support for various datasets built-in including VQA, VizWiz, TextVQA, VisualDialog and COCO Captioning.
- **Modules**: Provides implementations for many commonly used layers in vision and language domain
- **Distributed**: Support for distributed training based on DataParallel as well as DistributedDataParallel.
- **Unopinionated**: Unopinionated about the dataset and model implementations built on top of it.
- **Customization**: Custom losses, metrics, scheduling, optimizers, tensorboard; suits all your custom needs.

You can use Pythia to **_bootstrap_** for your next vision and language multimodal research project.

Pythia can also act as **starter codebase** for challenges around vision and
language datasets (TextVQA challenge, VQA challenge)

![Pythia Examples](https://i.imgur.com/BP8sYnk.jpg)

## Documentation

Learn more about Pythia [here](https://learnpythia.readthedocs.io/en/latest/).

## Training

Once we have the data downloaded and in place, we just need to select a model, an appropriate task and dataset as well related config file. Default configurations can be found  inside `configs` folder in repository's root folder. Configs are divided for models in format of `[task]/[dataset_key]/[model_key].yml` where `dataset_key` can be retrieved from the table above. For example, for `pythia` model, configuration for VQA 2.0 dataset can be found at `configs/vqa/vqa2/pythia.yml`. Following table shows the keys and the datasets
supported by the models in Pythia's model zoo.

| Model  | Key | Supported Datasets    | Pretrained Models | Notes                                                     |
|--------|-----------|-----------------------|-------------------|-----------------------------------------------------------|
| Pythia | pythia    | vqa2, vizwiz, textvqa, visual_genome | [vqa2 train+val](https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia_train_val.pth), [vqa2 train only](https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia.pth), [vizwiz](https://dl.fbaipublicfiles.com/pythia/pretrained_models/vizwiz/pythia_pretrained_vqa2.pth)  | VizWiz model has been pretrained on VQAv2 and transferred |
| LoRRA  | lorra     | vqa2, vizwiz, textvqa       | [textvqa](https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/lorra_best.pth)      |                               |
| CNN LSTM  | cnn_lstm     | clevr       |       | Features are calculated on fly. |                             
| BAN    | ban       | vqa2, vizwiz, textvqa | Coming soon!      | Support is preliminary and haven't been tested thoroughly. |
| BUTD    | butd       | coco | [coco](https://dl.fbaipublicfiles.com/pythia/pretrained_models/coco_captions/butd.pth)    |              |


For running `LoRRA` on `TextVQA`, run the following command from root directory of your pythia clone:

```
cd ~/pythia
python tools/run.py --tasks vqa --datasets textvqa --model lorra --config configs/vqa/textvqa/lorra.yml 
```

**Note for BUTD model :**  for training BUTD model use the config `butd.yml`. Training uses greedy decoding for validation. Currently we do not have support to train the model using beam search decoding validation. We will add that support soon. For inference only use `butd_beam_search.yml` config that supports beam search decoding.

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

The table below shows inference metrics for various pretrained models:

| Model  | Dataset          | Metric                     | Notes                         |
|--------|------------------|----------------------------|-------------------------------|
| Pythia | vqa2 (train+val) | test-dev accuracy - 68.31% | Can be easily pushed to 69.2% |
| Pythia | vqa2 (train)     | test-dev accuracy - 66.70%  |  |
| Pythia | vizwiz (train)     | test-dev accuracy - 54.22%  |    Pretrained on VQA2 and transferred to VizWiz                           |
| LoRRA  | textvqa (train)  | val accuracy - 27.4%       |                               |
| BUTD  | coco  (karpathy train)  | BLEU 1 - 76.02, BLEU 4 - 35.42 , METEOR - 27.39, ROUGE_L - 56.17, CIDEr - 112.03 , SPICE -  20.33    |   With Beam Search(length 5), Karpathy test split                           |

**Note** that, for simplicity, our current released model **does not** incorporate extensive data augmentations (e.g. visual genome, visual dialogue) during training, which was used in our challenge winner entries for VQA and VizWiz 2018. As a result, there can be some performance gap to models reported and released previously. If you are looking for reproducing those results, please checkout the [v0.1](https://github.com/facebookresearch/pythia/releases/tag/v0.1) release.

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

## License

Pythia is licensed under BSD license available in [LICENSE](LICENSE) file
