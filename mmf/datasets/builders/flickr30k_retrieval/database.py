# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import jsonlines
import torch
import random
import numpy as np
import _pickle as cPickle


class Flickr30kRetrievalDatabase(torch.utils.data.Dataset):
    def __init__(self, imdb_path, dataset_type, test_id_file_path, hard_neg_file_path):
        super().__init__()
        self._dataset_type = dataset_type
        self._load_annotations(imdb_path, test_id_file_path, hard_neg_file_path)
        self._metadata = {}

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, x):
        self._metadata = x

    def _load_annotations(self, imdb_path, test_id_path, hard_neg_file_path):

        with jsonlines.open(imdb_path) as reader:

            # Build an index which maps image id with a list of caption annotations.
            entries = []
            imgid2entry = {}
            count = 0

            remove_ids = []

            if test_id_path:
                remove_ids = np.load(test_id_path)
                remove_ids = [int(x) for x in remove_ids]

            for annotation in reader:
                image_id = int(annotation["img_path"].split(".")[0])

                if self._dataset_type == "train" and int(image_id) in remove_ids:
                    continue

                imgid2entry[image_id] = []
                for sentences in annotation["sentences"]:
                    entries.append({"caption": sentences, "image_id": image_id})
                    imgid2entry[image_id].append(count)
                    count += 1

        self._entries = entries
        self.imgid2entry = imgid2entry
        self.image_id_list = [*self.imgid2entry]

        if self._dataset_type == "train":
            with open(hard_neg_file_path, "rb") as f:
                image_info = cPickle.load(f)

            for key, value in image_info.items():
                setattr(self, key, value)

            self.train_imgId2pool = {
                imageId: i for i, imageId in enumerate(self.train_image_list)
            }

        self.db_size = len(self._entries)

    def __len__(self):
        return self.db_size

    def __getitem__(self, idx):

        entry = self._entries[idx]
        image_id = entry["image_id"]

        while True:
            # sample a random image:
            img_id2 = random.choice(self.image_id_list)
            if img_id2 != image_id:
                break

        entry2 = self._entries[random.choice(self.imgid2entry[img_id2])]

        # random image wrong
        while True:
            # sample a random image:
            img_id3 = random.choice(self.image_id_list)
            if img_id3 != image_id:
                break

        entry3 = self._entries[random.choice(self.imgid2entry[img_id3])]

        if self._dataset_type == "train":
            # random hard caption.
            rand_img_id_pool = self.train_hard_pool[self.train_imgId2pool[image_id]]
            pool_img_idx = int(
                rand_img_id_pool[np.random.randint(1, len(rand_img_id_pool))]
            )
            img_id4 = self.train_image_list[pool_img_idx]
        else:
            while True:
                # sample a random image:
                img_id4 = random.choice(self.image_id_list)
                if img_id4 != image_id:
                    break

        entry4 = self._entries[random.choice(self.imgid2entry[img_id4])]

        return [entry, entry2, entry3, entry4]
