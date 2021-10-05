# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/ChenRocks/UNITER/
# and adapted for MMF.

import torch
from mmf.utils.build import build_encoder
from torch import nn
from transformers.modeling_bert import BertPooler


class UniterModel(nn.Module):
    """ Modification for Joint Vision-Language Encoding
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = build_encoder(config.text_embeddings)
        self.img_embeddings = build_encoder(config.image_embeddings)
        self.encoder = build_encoder(config.encoder)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def init_weights(self, _):
        pass

    def _compute_txt_embeddings(self, input_ids, position_ids, txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(
        self, img_feat, img_pos_feat, img_masks=None, img_type_ids=None
    ):
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.embeddings.token_type_embeddings(img_type_ids)
        output = self.img_embeddings(
            img_feat, img_pos_feat, img_type_embeddings, img_masks
        )
        return output

    def _compute_img_txt_embeddings(
        self,
        input_ids,
        position_ids,
        img_feat,
        img_pos_feat,
        gather_index,
        img_masks=None,
        txt_type_ids=None,
        img_type_ids=None,
    ):
        txt_emb = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_pos_feat, img_masks, img_type_ids
        )
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size
        )
        embedding_output = torch.gather(
            torch.cat([txt_emb, img_emb], dim=1), dim=1, index=gather_index
        )
        return embedding_output

    def forward(
        self,
        input_ids,
        position_ids,
        img_feat,
        img_pos_feat,
        attention_mask,
        gather_index=None,
        img_masks=None,
        output_all_encoded_layers=True,
        txt_type_ids=None,
        img_type_ids=None,
    ):
        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids
            )
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids
            )
        else:
            embedding_output = self._compute_img_txt_embeddings(
                input_ids,
                position_ids,
                img_feat,
                img_pos_feat,
                gather_index,
                img_masks,
                txt_type_ids,
                img_type_ids,
            )

        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers
