# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch


def text_token_overlap_with_bbox(traces, bboxes, num_samples):
    np_traces = np.array(traces)
    tracex = np_traces[:, :, 0][..., None]  # of shape: [num_wp, num_samples_per_wp, 1]
    tracey = np_traces[:, :, 1][..., None]

    bboxes = np.array(bboxes)  # of shape: (num_bboxes, 5)
    num_bboxes = bboxes.shape[0]
    bboxes_tl_x = bboxes[:, 0].reshape(1, 1, num_bboxes)
    bboxes_tl_y = bboxes[:, 1].reshape(1, 1, num_bboxes)
    bboxes_br_x = bboxes[:, 2].reshape(1, 1, num_bboxes)
    bboxes_br_y = bboxes[:, 3].reshape(1, 1, num_bboxes)

    in_x = np.logical_and(tracex >= bboxes_tl_x, tracex <= bboxes_br_x)
    in_y = np.logical_and(tracey >= bboxes_tl_y, tracey <= bboxes_br_y)
    percent_overlaps = np.mean(np.logical_and(in_x, in_y), axis=1)
    return torch.FloatTensor(percent_overlaps).squeeze(1)
