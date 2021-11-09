# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/ChenRocks/UNITER/
# and adapted for MMF.

from typing import Optional

import torch
from mmf.utils.general import retry_n
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import BertConfig, BertEmbeddings, BertModel


NUM_RETRIES = 6
EMPTY_CONFIG = OmegaConf.create({})


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


class UNITERModelBase(nn.Module):
    """ UNITER embedding and transformer trunk for V-L modeling.
    Modified from https://github.com/ChenRocks/UNITER/ for MMF.
    https://arxiv.org/pdf/1909.11740.pdf

    By default, this model uses the pretrained bert-base-uncased
    transformer trunk with from huggingface.

    To train on this model through MMF, look at the UNTIER model,
    which supports pretraining and finetuning of UNITERModelBase
    with configurable heads.

    For an example of using this model standalone,
    take a look at its unit test in `test_uniter.py`.
    """

    def __init__(
        self,
        random_init: bool = False,
        bert_model_name: str = "bert-base-uncased",
        img_dim: int = 2048,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0,
        text_embeddings: DictConfig = EMPTY_CONFIG,
        encoder: DictConfig = EMPTY_CONFIG,
    ):
        super().__init__()

        bert_config = retry_n(
            NUM_RETRIES,
            BertConfig.from_pretrained,
            bert_model_name,
            **OmegaConf.to_container(text_embeddings),
        )
        self.text_embeddings = BertEmbeddings(bert_config)

        self.img_embeddings = UNITERImageEmbeddings(
            img_dim=img_dim,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
        )

        bert_model_name = bert_model_name
        hf_config = retry_n(
            NUM_RETRIES,
            BertConfig.from_pretrained,
            bert_model_name,
            **OmegaConf.to_container(encoder),
        )
        if random_init:
            bert_model = BertModel(hf_config)
        else:
            bert_model = retry_n(
                NUM_RETRIES,
                BertModel.from_pretrained,
                bert_model_name,
                config=hf_config,
            )
        self.encoder = bert_model.encoder
        self.pooler = bert_model.pooler

    def _compute_txt_embeddings(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        output = self.text_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        return output

    def _compute_img_embeddings(
        self,
        img_feat: Tensor,
        img_pos_feat: Tensor,
        img_masks: Optional[Tensor] = None,
        img_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.text_embeddings.token_type_embeddings(img_type_ids)
        output = self.img_embeddings(
            img_feat, img_pos_feat, img_type_embeddings, img_masks
        )
        return output

    def _compute_img_txt_embeddings(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        img_feat: Tensor,
        img_pos_feat: Tensor,
        img_masks: Optional[Tensor] = None,
        txt_type_ids: Optional[Tensor] = None,
        img_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        txt_emb = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_pos_feat, img_masks, img_type_ids
        )
        embedding_output = torch.cat([txt_emb, img_emb], dim=1)
        return embedding_output

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        img_feat: Tensor,
        img_pos_feat: Tensor,
        attention_mask: Tensor,
        img_masks: Optional[Tensor] = None,
        output_hidden_states: bool = False,
        txt_type_ids: Optional[Tensor] = None,
        img_type_ids: Optional[Tensor] = None,
        input_modality: str = "image-text",
    ) -> Tensor:
        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        # https://github.com/huggingface/transformers/issues/542 for details
        # on why we add very negative values to attention scores
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_modality == "image":
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids
            )
        elif input_modality == "text":
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
                img_masks,
                txt_type_ids,
                img_type_ids,
            )

        encoded_layers = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=output_hidden_states,
        )
        if not output_hidden_states:
            encoded_layers = encoded_layers[-1]
        return encoded_layers
