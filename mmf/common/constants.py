# Copyright (c) Facebook, Inc. and its affiliates.
imdb_version = 1
FASTTEXT_WIKI_URL = (
    "https://dl.fbaipublicfiles.com/pythia/pretrained_models/fasttext/wiki.en.bin"
)

CLEVR_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"

VISUAL_GENOME_CONSTS = {
    "imdb_url": "https://dl.fbaipublicfiles.com/pythia/data/imdb/visual_genome.tar.gz",
    "features_url": "https://dl.fbaipublicfiles.com/pythia/features/visual_genome.tar.gz",  # noqa
    "synset_file": "vg_synsets.txt",
    "vocabs": "https://dl.fbaipublicfiles.com/pythia/data/vocab.tar.gz",
}

VISUAL_DIALOG_CONSTS = {
    "imdb_url": {
        "train": "https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip?dl=1",
        "val": "https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=1",
        "test": "https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_test.zip?dl=1",
    },
    "features_url": {
        "visual_dialog": "https://dl.fbaipublicfiles.com/pythia/features/visual_dialog.tar.gz",  # noqa
        "coco": "https://dl.fbaipublicfiles.com/pythia/features/coco.tar.gz",
    },
    "vocabs": "https://dl.fbaipublicfiles.com/pythia/data/vocab.tar.gz",
}

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
