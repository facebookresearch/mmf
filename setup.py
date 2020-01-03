#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
import os.path
import shutil
from glob import glob
import sys

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext

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

ext_modules = [
    Extension(
        'cphoc',
        sources=['pythia/utils/phoc/src/cphoc.c'],
        language='c',
        libraries=["pthread", "dl", "util", "rt", "m"],
        extra_compile_args=["-O3"],
    ),
]


class BuildExt(build_ext):
    def run(self):
        build_ext.run(self)
        cphoc_lib = glob('build/lib.*/cphoc.*.so')[0]
        shutil.copy(cphoc_lib, 'pythia/utils/phoc/cphoc.so')


if __name__ == "__main__":
    setuptools.setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        packages=setuptools.find_packages(),
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExt},
        version="0.3",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        license=LICENSE,
        setup_requires=["pytest-runner"],
        tests_require=["flake8", "pytest"],
    )
