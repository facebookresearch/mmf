# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/ChenRocks/UNITER/
# and adapted for MMF.

from typing import Optional

from torch import Tensor, nn


class UNITERImageEmbeddings(nn.Module):
    """
    Image Embeddings used by UNITER.
    Code modified from https://github.com/ChenRocks/UNITER/blob/master/model/model.py
    Performs a linear projection then normalization over image and position features.
    """

    def __init__(
        self,
        img_dim: int = 2048,
        hidden_size: int = 768,
        eps: float = 1e-12,
        hidden_dropout_prob: float = 0,
        pos_dim: int = 7,
    ):
        super().__init__()

        self.img_linear = nn.Linear(img_dim, hidden_size)
        self.img_layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.pos_layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.pos_linear = nn.Linear(pos_dim, hidden_size)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        img_feat: Tensor,
        img_pos_feat: Tensor,
        type_embeddings: Tensor,
        img_masks: Optional[Tensor] = None,
    ) -> Tensor:
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.final_layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
