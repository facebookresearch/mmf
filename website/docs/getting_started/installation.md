---
id: installation
title: Installation
sidebar_label: Installation
---

MMF is tested on Python 3.7+ and PyTorch 1.5.

## Install using pip

MMF can be installed from pip with following command:

```bash
pip install --upgrade --pre mmf
```

Use this if:

- You are using MMF as a library and not developing inside MMF. Have a look at extending MMF tutorial.
- You want easy installation and don't care about up-to-date features. Note that, pip packages are always outdated compared to installing from source.

## Install from source

To install from source, do:

```bash
git clone https://github.com/facebookresearch/mmf.git
cd mmf
pip install --editable .
```

## Running tests

MMF uses pytest for testing purposes. To ensure everything and run tests at your end do:

```bash
pytest ./tests/
```
