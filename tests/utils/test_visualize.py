# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import numpy as np
import torch
from mmf.utils.features.visualizing_image import SingleImageViz


class TestVisualize(unittest.TestCase):
    objids = np.array(["obj0", "obj1", "obj2", "obj3"])
    attrids = np.array(["attr0", "attr1", "attr2", "attr3"])
    img = np.array(
        [
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
            [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
            [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
            [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11]],
        ],
        dtype=np.uint8,
    )
    output_dict = {
        "obj_ids": torch.tensor([0, 1]),
        "obj_probs": torch.tensor([0.5, 0.25]),
        "attr_ids": torch.tensor([2, 3]),
        "attr_probs": torch.tensor([0.3, 0.6]),
        "boxes": torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]]),
    }

    buffer = np.array(
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [62, 48, 70],
                [112, 87, 124],
                [53, 41, 58],
                [38, 30, 42],
                [28, 22, 31],
            ],
            [
                [6, 6, 6],
                [3, 3, 3],
                [3, 3, 3],
                [4, 4, 4],
                [4, 4, 4],
                [4, 4, 4],
                [5, 5, 5],
            ],
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
            ],
            [
                [6, 6, 6],
                [7, 7, 7],
                [8, 8, 8],
                [9, 9, 9],
                [9, 9, 9],
                [10, 10, 10],
                [11, 11, 11],
            ],
            [
                [6, 6, 6],
                [7, 7, 7],
                [8, 8, 8],
                [9, 9, 9],
                [9, 9, 9],
                [10, 10, 10],
                [11, 11, 11],
            ],
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
            ],
            [
                [6, 6, 6],
                [7, 7, 7],
                [8, 8, 8],
                [9, 9, 9],
                [9, 9, 9],
                [10, 10, 10],
                [11, 11, 11],
            ],
        ]
    )

    def test_single_image_viz(self) -> None:
        frcnn_visualizer = SingleImageViz(
            self.img, id2obj=self.objids, id2attr=self.attrids
        )
        frcnn_visualizer.draw_boxes(
            self.output_dict.get("boxes"),
            self.output_dict.pop("obj_ids"),
            self.output_dict.pop("obj_probs"),
            self.output_dict.pop("attr_ids"),
            self.output_dict.pop("attr_probs"),
        )

        buffer = frcnn_visualizer._get_buffer()

        self.assertTrue((buffer == self.buffer).all())
