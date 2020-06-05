---
id: concepts
title: Terminology and Concepts
sidebar_label: Terminology and Concepts
---

**[Outdated]** A new version of this will be uploaded soon

To develop on top of MMF, it is necessary to understand concepts and terminology used in MMF codebase. MMF has been very carefully designed from ground-up to be a multi-tasking framework. This means using MMF you can train on multiple datasets/datasets together.

To achieve this, MMF has few opinions about architecture of your research project. But, being generic means MMF abstracts a lot of concepts in its modules and it would be easy to develop on top of MMF once a developer understands these simple concepts. Major concepts and terminology in MMF that one needs to know in order to develop over MMF are as follows:

- [Tasks and Datasets](#tasks-and-datasets)
- [Models](#models)
- [Registry](#registry)
- [Configuration](#configuration)
- [Processors](#processors)
- [Sample List](#sample-list)

## Tasks and Datasets

In MMF, we have divided datasets into a set category of tasks. Thus, a task corresponds to a collection of datasets that belong to it. For example, VQA 2.0, VizWiz and TextVQA all belong VQA task. Each task and dataset has been assigned a unique key which is used to refer it in the command line arguments.

Following table shows the tasks and their datasets:

|         |            |                                              |
| ------- | ---------- | -------------------------------------------- |
| Task    | Key        | Datasets                                     |
| VQA     | vqa        | VQA2.0, VizWiz, TextVQA, VisualGenome, CLEVR |
| Dialog  | dialog     | VisualDialog                                 |
| Caption | captioning | MS COCO                                      |

Following table shows the inverse of the above table, datasets along with their tasks and keys:

|              |               |            |              |
| ------------ | ------------- | ---------- | ------------ |
| Datasets     | Key           | Task       | Notes        |
| VQA 2.0      | vqa2          | vqa        |              |
| TextVQA      | textvqa       | vqa        |              |
| VizWiz       | vizwiz        | vqa        |              |
| VisualDialog | visdial       | dialog     | Coming soon! |
| VisualGenome | visual_genome | vqa        |              |
| CLEVR        | clevr         | vqa        |              |
| MS COCO      | coco          | captioning |              |

## Models

Reference implementations for state-of-the-art models have been included to act as a base for reproduction of research papers and starting point of new research. MMF has been used in past for following papers:

- [Towards VQA Models That Can Read (LoRRA model)](https://arxiv.org/abs/1904.08920)
- [VQA 2018 Challenge winner](https://arxiv.org/abs/1807.09956)
- VizWiz 2018 Challenge winner

Similar to tasks and datasets, each model has been registered with a unique key for easy reference in configuration and command line arguments. Following table shows each model's key name and datasets it can be run on.

|          |          |                                      |
| -------- | -------- | ------------------------------------ |
| Model    | Key      | Datasets                             |
| LoRRA    | lorra    | vqa2, textvqa, vizwiz                |
| Pythia   | pythia   | textvqa, vizwiz, vqa2, visual_genome |
| BAN      | ban      | textvqa, vizwiz, vqa2                |
| BUTD     | butd     | coco                                 |
| CNN LSTM | cnn_lstm | clevr                                |

:::note

BAN support is preliminary and hasn't been properly fine-tuned yet.

:::

## Registry

Registry acts as a central source of truth for MMF. Inspired from Redux's global store, useful information needed by MMF ecosystem is registered in the `registry`. Registry can be considered as a general purpose storage for information which is needed by multiple parts of the framework and acts source of information wherever that information is needed.

Registry also registers models, tasks, datasets etc. based on a unique key as mentioned above. Registry's functions can be used as decorators over the classes which need to be registered (for e.g. models etc.)

Registry object can be imported as the follow:

```
from mmf.common.registry import registry

```

Find more details about Registry class in its documentation [common/registry](../lib/common/registry).

## Configuration

As is necessary with research, most of the parameters/settings in MMF are configurable. MMF specific default values (`training`) are present in [mmf/common/defaults/configs/base.yaml](https://github.com/facebookresearch/mmf/blob/v0.3/mmf/common/defaults/configs/base.yaml) with detailed comments delineating the usage of each parameter.

For ease of usage and modularity, configuration for each dataset is kept separately in `mmf/common/defaults/configs/datasets/[task]/[dataset].yaml` where you can get `[task]` value for the dataset from the tables in [Tasks and Datasets](#tasks-and-datasets) section.

The most dynamic part, model configuration are also kept separate and are the one which need to be defined by the user if they are creating their own models. We include configurations for the models included in the model zoo of MMF. For each model, there is a separate configuration for each dataset it can work on. See an example in [configs/vqa/vqa2/pythia.yaml](https://github.com/facebookresearch/mmf/blob/v0.3/configs/vqa/vqa2/pythia.yaml). The configuration in the configs folder are divided using the scheme `configs/[task]/[dataset]/[model].yaml`.

It is possible to include other configs into your config using `includes` directive. Thus, in MMF config above you can include `vqa2`'s config like this:

```
includes:
- common/defaults/configs/datasets/vqa/vqa2.yaml
```

Now, due to separate config per dataset this concept can be extended to do multi-tasking and include multiple dataset configs here.

`base.yaml` file mentioned above is always included and provides sane defaults for most of the training parameters. You can then specify the config of the model that you want to train using `--config [config_path]` option. The final config can be retrieved using `registry.get('config')` anywhere in your codebase. You can access the attributes from these configs by using `dot` notation. For e.g. if you want to get the value of maximum iterations, you can get that by `registry.get('config').training.max_updates`.

The values in the configuration can be overriden using two formats:

- Individual Override: For e.g. you want to use `DataParallel` to train on multiple GPUs, you can override the default value of `False` by passing arguments `training.data_parallel True` at the end your command. This will override that option on the fly.
- DemJSON based override: The above option gets clunky when you are trying to run the hyperparameters sweeps over model parameters. To avoid this, you can update a whole block using a demjson string. For e.g. to use early stopping as well update the patience, you can pass `--config_override "{training: {should_early_stop: True, patience: 5000}}"`. This demjson string is easier to generate programmatically than the individual override.

:::tip

It is always helpful to verify your config overrides and final configuration values that are printed to make sure you override the correct keys.

:::

## Processors

The main aim of processors is to keep data processing pipelines as similar as possible for different datasets and allow code reusability. Processors take in a dict with keys corresponding to data they need and return back a dict with processed data. This helps keep processors independent of the rest of the logic by fixing the signatures they require. Processors are used in all of the datasets to hand off the data processing needs. Learn more about processors in the [documentation for processors](../lib/datasets/processors).

## Sample List

[SampleList](../lib/datasets/sample#mmf.common.sample.SampleList) has been inspired from BBoxList in maskrcnn-benchmark, but is more generic. All datasets integrated with MMF need to return a [Sample](../lib/datasets/sample#mmf.common.sample.Sample) which will be collated into `SampleList`. Now, `SampleList` comes with a lot of handy functions which enable easy batching and access of things. For e.g. `Sample` is a dict with some keys. In `SampleList`, values for these keys will be smartly clubbed based on whether it is a tensor or a list and assigned back to that dict. So, end user gets these keys clubbed nicely together and can use them in their model. Models integrated with Pythia receive a `SampleList` as an argument which again makes the trainer unopinionated about the models as well as the datasets. Learn more about `Sample` and `SampleList` in their [documentation](../lib/common/sample).
