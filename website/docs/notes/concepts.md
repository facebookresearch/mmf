---
id: concepts
title: Terminology and Concepts
sidebar_label: Terminology and Concepts
---

To develop on top of MMF, it is necessary to understand concepts and terminology used in MMF codebase. MMF has been very carefully designed from ground-up to be a multi-tasking framework. This means using MMF you can train on multiple datasets/datasets together.

To achieve this, MMF has few opinions about architecture of your research project. But, being generic means MMF abstracts a lot of concepts in its modules and it would be easy to develop on top of MMF once a developer understands these simple concepts. Major concepts and terminology in MMF that one needs to know in order to develop over MMF are as follows:

- [Datasets](#datasets)
- [Models](#models)
- [Registry](#registry)
- [Configuration](#configuration)
- [Processors](#processors)
- [Sample List](#sample-list)

## Datasets

You can find all the latest datasets [here](https://github.com/facebookresearch/mmf/tree/master/mmf/configs/datasets).

The dataset's key is available under the particular dataset's config, ie., for vizwiz's key, you can look in vizwiz's config available [here](https://github.com/facebookresearch/mmf/blob/master/mmf/configs/datasets/vizwiz/defaults.yaml)

```yaml
dataset_config:
  vizwiz: # dataset key
      data_dir: ${env.data_dir}/datasets
      depth_first: false
      fast_read: false
      use_images: false
      use_features: true
      zoo_requirements:
      - vizwiz.v2019
      ...
```

## Models

Reference implementations for state-of-the-art models have been included to act as a base for reproduction of research papers and starting point of new research. MMF has been used in past for following papers:

- [Towards VQA Models That Can Read (LoRRA model)](https://arxiv.org/abs/1904.08920)
- [VQA 2018 Challenge winner](https://arxiv.org/abs/1807.09956)
- [VizWiz 2018 Challenge winner](https://vizwiz.org/wp-content/uploads/2019/06/workshop2018_slides_FAIR_A-STAR.pdf)
- [VQA 2020 Challenge winner](https://github.com/facebookresearch/mmf/tree/master/projects/movie_mcan)

Similar to datasets, each model has been registered with a unique key for easy reference in configuration and command line arguments. For a more complete list of models, please see [here](https://github.com/facebookresearch/mmf/tree/master/mmf/configs/models)

The model's key is available under the particular model's config, ie., for mmf_transformer, the model's config file is available under [here](https://github.com/facebookresearch/mmf/blob/master/mmf/configs/models/mmf_transformer/defaults.yaml)

```yaml
model_config:
  mmf_transformer: # model key
    transformer_base: bert-base-uncased
    training_head_type: classification
    backend:
      type: huggingface
      freeze: false
      params: {}
    ...
```

## Registry

Registry acts as a central source of truth for MMF. Inspired from Redux's global store, useful information needed by MMF ecosystem is registered in the `registry`. Registry can be considered as a general purpose storage for information which is needed by multiple parts of the framework and acts source of information wherever that information is needed.

Registry also registers models, tasks, datasets etc. based on a unique key as mentioned above. Registry's functions can be used as decorators over the classes which need to be registered (for e.g. models etc.)

Registry object can be imported as the follow:

```
from mmf.common.registry import registry

```

Find more details about Registry class in its documentation [common/registry](https://mmf.sh/api/lib/common/registry.html).

## Configuration

As is necessary with research, most of the parameters/settings in MMF are configurable. MMF specific default values (`training`) are present in [mmf/configs/defaults.yaml](https://github.com/facebookresearch/mmf/blob/master/mmf/configs/defaults.yaml) with detailed comments delineating the usage of each parameter.

For ease of usage and modularity, configuration for each dataset is kept separately in `mmf/configs/datasets/[dataset]/[variants].yaml` where you can get `[dataset]` value for the dataset from the tables in [Datasets](#datasets) section.

The most dynamic part, model configurations are also kept separate and are the ones which need to be defined by the user if they are creating their own models. We include configurations for the models included in the model zoo of MMF. You can find the model configurations [here](https://github.com/facebookresearch/mmf/tree/master/mmf/configs/models)


It is possible to include other configs into your config using `includes` directive. Thus, in MMF config above you can include `lxmert`'s config like this:

```yaml
includes:
- configs/models/lxmert/defaults.yaml
```

Now, due to separate config per dataset this concept can be extended to do multi-tasking and include multiple dataset configs here.

`defaults.yaml` file mentioned above is always included and provides sane defaults for most of the training parameters. You can then specify the config of the model that you want to train using `--config [config_path]` option. The final config can be retrieved using `registry.get('config')` anywhere in your codebase. You can access the attributes from these configs by using `dot` notation. For e.g. if you want to get the value of maximum iterations, you can get that by `registry.get('config').training.max_updates`.

The values in the configuration can be overriden using two formats:

- Individual Override: For e.g. you want to use `DataParallel` to train on multiple GPUs, you can override the default value of `False` by passing arguments `training.data_parallel True` at the end your command. This will override that option on the fly.
- DemJSON based override: The above option gets clunky when you are trying to run the hyperparameters sweeps over model parameters. To avoid this, you can update a whole block using a demjson string. For e.g. to use early stopping as well update the patience, you can pass `--config_override "{training: {should_early_stop: True, patience: 5000}}"`. This demjson string is easier to generate programmatically than the individual override.

:::tip

It is always helpful to verify your config overrides and final configuration values that are printed to make sure you override the correct keys.

:::

## Processors

The main aim of processors is to keep data processing pipelines as similar as possible for different datasets and allow code reusability. Processors take in a dict with keys corresponding to data they need and return back a dict with processed data. This helps keep processors independent of the rest of the logic by fixing the signatures they require. Processors are used in all of the datasets to hand off the data processing needs. Learn more about processors in the [documentation for processors](https://mmf.sh/api/lib/datasets/processors.html).

## Sample List

[SampleList](https://mmf.sh/api/lib/common/sample.html#mmf.common.sample.SampleList) has been inspired from BBoxList in maskrcnn-benchmark, but is more generic. All datasets integrated with MMF need to return a [Sample](https://mmf.sh/api/lib/common/sample.html#mmf.common.sample.Sample) which will be collated into `SampleList`. Now, `SampleList` comes with a lot of handy functions which enable easy batching and access of things. For e.g. `Sample` is a dict with some keys. In `SampleList`, values for these keys will be smartly clubbed based on whether it is a tensor or a list and assigned back to that dict. So, end user gets these keys clubbed nicely together and can use them in their model. Models integrated with MMF receive a `SampleList` as an argument which again makes the trainer unopinionated about the models as well as the datasets. Learn more about `Sample` and `SampleList` in their [documentation](https://mmf.sh/api/lib/common/sample.html).

:::tip

SampleList is a dict that works as a collator

:::
