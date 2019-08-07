# Copyright (c) Facebook, Inc. and its affiliates.
import os


imdb_version = 1
FASTTEXT_WIKI_URL = (
    "https://dl.fbaipublicfiles.com/pythia/pretrained_models/fasttext/wiki.en.bin"
)

CLEVR_DOWNLOAD_URL = (
    "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
)

VISUAL_GENOME_CONSTS = {
    "imdb_url": "https://dl.fbaipublicfiles.com/pythia/data/imdb/visual_genome.tar.gz",
    "features_url": "https://dl.fbaipublicfiles.com/pythia/features/visual_genome.tar.gz",
    "synset_file": "vg_synsets.txt"
}

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
