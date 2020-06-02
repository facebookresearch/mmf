#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

import codecs
import os
import platform
import re
from glob import glob

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext


def clean_html(raw_html):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html).strip()
    return cleantext


# Single sourcing code from here:
# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(*file_paths):
    here = os.path.abspath(os.path.dirname(__file__))

    def read(*parts):
        with codecs.open(os.path.join(here, *parts), "r") as fp:
            return fp.read()

    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def fetch_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
        # https://stackoverflow.com/a/12982689
        readme = clean_html(readme)
    return readme


def fetch_requirements():
    requirements_file = "requirements.txt"

    if platform.system() == "Windows":
        DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")

    with open(requirements_file) as f:
        reqs = f.read()

    reqs = reqs.strip().split("\n")
    reqs = remove_specific_requirements(reqs)
    return reqs


def remove_specific_requirements(reqs):
    rtd = "READTHEDOCS" in os.environ
    excluded = {"fasttext": rtd}
    updated_reqs = []
    for req in reqs:
        without_version = req.split("==")[0]
        if not excluded.get(without_version, False):
            updated_reqs.append(req)
    return updated_reqs


def fetch_files_from_folder(folder):
    options = glob(f"{folder}/**", recursive=True)
    data_files = []
    # All files inside the folder need to be added to package_data
    # which would include yaml configs as well as project READMEs
    for option in options:
        if os.path.isdir(option):
            files = []
            for f in glob(os.path.join(option, "*")):
                if os.path.isfile(f):
                    files.append(f)
                data_files += files
    return data_files


def fetch_package_data():
    current_dir = os.getcwd()
    mmf_folder = os.path.dirname(os.path.abspath(__file__))
    # The files for package data need to be relative to mmf package dir
    os.chdir(os.path.join(mmf_folder, "mmf"))
    data_files = fetch_files_from_folder("projects")
    data_files += fetch_files_from_folder("tools")
    data_files += fetch_files_from_folder("configs")
    data_files += glob(os.path.join("utils", "phoc", "cphoc.*"))
    os.chdir(current_dir)
    return data_files


DISTNAME = "mmf"
DESCRIPTION = "mmf: a modular framework for vision and language multimodal \
research."
LONG_DESCRIPTION = fetch_long_description()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
AUTHOR = "Facebook AI Research"
AUTHOR_EMAIL = "mmf@fb.com"
DEPENDENCY_LINKS = []
REQUIREMENTS = (fetch_requirements(),)
# Need to exclude folders in tests as well so as they don't create an extra package
# If something from tools is regularly used consider converting it into a cli command
EXCLUDES = ("data", "docs", "tests", "tests.*", "tools", "tools.*")
CMD_CLASS = {"build_ext": build_ext}
EXT_MODULES = [
    Extension(
        "mmf.utils.phoc.cphoc", sources=["mmf/utils/phoc/src/cphoc.c"], language="c"
    )
]

if "READTHEDOCS" in os.environ:
    # Don't build extensions when generating docs
    EXT_MODULES = []
    CMD_CLASS.pop("build_ext", None)
    # use CPU build of PyTorch
    DEPENDENCY_LINKS.append(
        "https://download.pytorch.org/whl/cpu/torch-1.5.0%2B"
        + "cpu-cp36-cp36m-linux_x86_64.whl"
    )


if __name__ == "__main__":
    setuptools.setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        include_package_data=True,
        package_data={"mmf": fetch_package_data()},
        packages=setuptools.find_packages(exclude=EXCLUDES),
        python_requires=">=3.6",
        ext_modules=EXT_MODULES,
        cmdclass=CMD_CLASS,
        version=find_version("mmf", "version.py"),
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        dependency_links=DEPENDENCY_LINKS,
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
        entry_points={
            "console_scripts": [
                "mmf_run = mmf_cli.run:run",
                "mmf_predict = mmf_cli.predict:predict",
                "mmf_convert_hm = mmf_cli.hm_convert:main",
            ]
        },
    )
