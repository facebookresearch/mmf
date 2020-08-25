# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Type, Union

import torch
from torch.nn import functional as F
from PIL import Image
from torch import nn

ImageType = Union[Type[Image.Image], str]


class RetrieverInterface(nn.Module):
    """Interface for MMBT Grid for Hateful Memes.
    """
    def __init__(self):
        super().__init__()

    def ref_encode_image(self, image: ImageType):
        raise NotImplementedError

    def ref_encode_text(self, input_ids, segment_ids):
        raise NotImplementedError

    def calc_batch_sim(self, rawQ_embeds, rawA_embeds):
        rawA_embeds = F.normalize(rawA_embeds)
        rawQ_embeds = F.normalize(rawQ_embeds)
        sim_scores = torch.matmul(rawQ_embeds, rawA_embeds.T)
        return sim_scores

    def convert_dims(self, embeds):
        if len(embeds.shape) != 2:
            if len(embeds.shape) == 1:
                embeds = embeds.unsqueeze(0)
            if len(embeds.shape) == 3:
                embeds = embeds.squeeze(1)

        assert len(embeds.shape) == 2
        return embeds
