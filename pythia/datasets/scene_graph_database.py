# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.datasets.image_database import ImageDatabase


class SceneGraphDatabase(ImageDatabase):
    def __init__(self, scene_graph_path):
        super().__init__(scene_graph_path)
        self.data_dict = {}
        for item in self.data:
            self.data_dict[item["image_id"]] = item

    def __getitem__(self, idx):
        return self.data_dict[idx]
