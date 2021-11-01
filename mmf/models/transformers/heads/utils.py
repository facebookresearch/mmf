# Copyright (c) Facebook, Inc. and its affiliates.

from torch import Tensor


def compute_masked_hidden(hidden: Tensor, mask: Tensor) -> Tensor:
    """ Get only the masked region.

    hidden: tensor, dim (bs, num_feat, feat_dim)
    mask: bool tensor, dim (bs, num_feat)
    Returns a tensor of dim (bs * num_feat_unmasked, feat_dim),
    containing the features in hidden that are True in the mask tensor.
    """
    mask = mask.unsqueeze(-1).expand_as(hidden)
    hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
    return hidden_masked
