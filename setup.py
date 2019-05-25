#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
import os.path
import sys

import setuptools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pythia"))

with open("README.md", encoding="utf8") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

with open("requirements.txt") as f:
    reqs = f.read()

DISTNAME = "pythia"
DESCRIPTION = "pythia: a modular framework for vision and language multimodal \
research."
LONG_DESCRIPTION = readme
AUTHOR = "Facebook AI Research"
LICENSE = license
REQUIREMENTS = (reqs.strip().split("\n"),)

if __name__ == "__main__":
    setuptools.setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        packages=setuptools.find_packages(),
        dependency_links=[
            "https://github.com/facebookresearch/fastText/tarball/master#egg=fastText"
        ],
        version="0.3",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        license=LICENSE,
        setup_requires=["pytest-runner"],
        tests_require=["flake8", "pytest"],
    )
