---
id: installation
title: Installation
sidebar_label: Installation
---

MMF has been tested on Python 3.7+ and PyTorch 1.6. We recommend using a conda environment to install MMF.

## Creating a conda environment [Optional]

Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, run:

```bash
conda create -n mmf python=3.7
conda activate mmf
```

## Install from source [Recommended]

To install from source do:

```bash
git clone https://github.com/facebookresearch/mmf.git
cd mmf
pip install --editable .
```

## Install using pip

MMF can be installed using pip with the following command:

```bash
pip install --upgrade --pre mmf
```

Use this if:

- You are using MMF as a library and not developing inside MMF. Take a look at the extending MMF tutorial.
- You want easy installation and don't care about up-to-date features. Note that pip packages are always outdated relative to installing from source.

Alternatively, to install latest MMF version from GitHub using pip, use

```bash
pip install git+https://github.com/facebookresearch/mmf.git
```

## Windows

If you are on Windows, run the pip install commands with an extra argument like:

```
pip install -f https://download.pytorch.org/whl/torch_stable.html --editable .
```

## Running tests [Optional]

MMF uses pytest for testing. To verify everything and run tests at your end do:

```bash
pytest ./tests/
```


## Contributing to MMF

We welcome all contributions to MMF. Have a look at our [contributing guidelines](https://github.com/facebookresearch/mmf/tree/master/.github/CONTRIBUTING.md) to get started.
