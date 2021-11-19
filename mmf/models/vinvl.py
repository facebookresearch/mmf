# Copyright (c) Facebook, Inc. and its affiliates.

# Code based off https://github.com/microsoft/Oscar
# modified for MMF
# Licensed under the MIT license.

import logging
from collections import namedtuple
from typing import Dict, Optional, Tuple

import torch
from mmf.common.sample import SampleList
from mmf.models.transformers.heads.mlm import MLM
from mmf.models.transformers.heads.mlp import MLP
from mmf.utils.general import retry_n
from omegaconf import OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
    BertConfig,
)

logger = logging.getLogger(__name__)

EMPTY_CONFIG = OmegaConf.create({})
NUM_RETRIES = 6


class BertImgModel(BertPreTrainedModel):
    """VinVL Bert Encoder for image features
    From https://github.com/microsoft/Oscar/blob/master/oscar/modeling/modeling_bert.py
    Is a thin wrapper around BertEncoder that handles image features
    """

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.img_dim = config.img_feature_dim
        self.use_img_layernorm = getattr(config, "use_img_layernorm", False)

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = nn.LayerNorm(
                config.hidden_size, eps=config.img_layer_norm_eps
            )

    def forward(
        self,
        input_ids: Tensor,
        img_feats: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.size(0), input_ids.size(1) + img_feats.size(1))
            ).to(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular
        # masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = self._get_head_mask(head_mask)

        # Do embeddings
        text_embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        img_embedding_output = self.img_embedding(img_feats)
        if self.use_img_layernorm:
            img_embedding_output = self.LayerNorm(img_embedding_output)
        img_embedding_output = self.dropout(img_embedding_output)
        embedding_output = torch.cat((text_embedding_output, img_embedding_output), 1)

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_hidden_states=True,
        )
        layers = namedtuple("TransformerOutput", ["final_layer", "hidden_layers"])
        return layers(encoder_outputs[0], encoder_outputs[1])

    def _get_head_mask(self, head_mask):
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        return head_mask


class VinVLForClassification(nn.Module):
    """VINVL wrapper for classification"""

    def __init__(
        self,
        mlp_config: Optional[Dict] = None,
        loss_config: Optional[Dict] = None,
        random_init: bool = False,
        bert_model_name: str = "bert-base-uncased",
        img_feature_dim: int = 2054,
        use_img_layernorm: bool = True,
        img_layer_norm_eps: float = 1e-12,
        *args,
        **kwargs,
    ):
        super().__init__()
        bert_config = BertConfig.from_pretrained(bert_model_name)
        # augment hf BertConfig for vinvl BertImgModel config
        bert_config.img_feature_dim = img_feature_dim
        bert_config.use_img_layernorm = use_img_layernorm
        bert_config.img_layer_norm_eps = img_layer_norm_eps

        if random_init:
            self.bert = BertImgModel(bert_config)
        else:
            self.bert = retry_n(
                NUM_RETRIES,
                BertImgModel.from_pretrained,
                bert_model_name,
                config=bert_config,
            )

        if mlp_config is None:
            mlp_config = {"num_layers": 0}
        if loss_config is None:
            loss_config = {}
        self.classifier = MLP(config=mlp_config)
        self.ce_loss = nn.CrossEntropyLoss(**loss_config)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        img_feats: Tensor,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        sequence_output = self.bert(
            input_ids,
            img_feats=img_feats,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        ).final_layer
        logits = self.classifier(sequence_output)["scores"]
        result = {"scores": logits}

        if labels is not None:
            ce_loss = self.ce_loss(logits.view(-1, logits.size(1)), labels.view(-1))
            result["losses"] = {"ce": ce_loss}
        return result


class VinVLForPretraining(nn.Module):
    """VINVL wrapper for pretraining
    MLM loss is described in https://arxiv.org/pdf/2004.06165.pdf
    Contrastive loss is an itm loss to guess,
        0 for a match,
        1 for a corrupt caption,
        2 for corrupt image labels
    VinVL trains with object detection labels concatanated with the input text.
    """

    def __init__(
        self,
        mlm_config: Optional[Dict] = None,
        contrast_config: Optional[Dict] = None,
        random_init: bool = False,
        bert_model_name: str = "bert-base-uncased",
        img_feature_dim: int = 2054,
        use_img_layernorm: bool = True,
        img_layer_norm_eps: float = 1e-12,
        *args,
        **kwargs,
    ):
        super().__init__()
        bert_config = retry_n(
            NUM_RETRIES,
            BertConfig.from_pretrained,
            bert_model_name,
        )
        # augment hf BertConfig for vinvl BertImgModel config
        bert_config.img_feature_dim = img_feature_dim
        bert_config.use_img_layernorm = use_img_layernorm
        bert_config.img_layer_norm_eps = img_layer_norm_eps

        if random_init:
            self.bert = BertImgModel(bert_config)
        else:
            self.bert = retry_n(
                NUM_RETRIES,
                BertImgModel.from_pretrained,
                bert_model_name,
                config=bert_config,
            )

        if mlm_config is None:
            mlm_config = {}
        if contrast_config is None:
            contrast_config = {"num_layers": 0, "num_labels": 3}
        self.mlm_head = MLM(config=mlm_config)
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrast_head = MLP(config=contrast_config)

    def mlm_forward(
        self,
        input_ids_masked: Tensor,
        lm_label_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        img_feats: Tensor,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        hidden_layers = self.bert(
            input_ids_masked,
            img_feats=img_feats,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        ).final_layer

        mlm_labels = {}
        mlm_labels["text"] = lm_label_ids
        mlm_labels["image"] = torch.full(
            img_feats.shape[:2],
            fill_value=-1,
            dtype=torch.long,
            device=lm_label_ids.device,
        )
        mlm_labels["combined_labels"] = torch.cat(
            [mlm_labels["text"], mlm_labels["image"]], dim=-1
        )

        processed_sample_list = SampleList({"mlm_labels": mlm_labels})
        return self.mlm_head(hidden_layers, processed_sample_list=processed_sample_list)

    def contrastive_forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        img_feats: Tensor,
        contrastive_labels: Tensor,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        final_layer = self.bert(
            input_ids,
            img_feats=img_feats,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        ).final_layer
        logits = self.contrast_head(final_layer)["scores"]
        loss = self.ce_loss(
            logits.contiguous().view(-1, 3),
            contrastive_labels.contiguous().view(-1),
        )
        return {"vinvl_three_way_contrastive": loss}

    def forward(
        self,
        input_ids_masked: Tensor,
        input_ids_corrupt: Tensor,
        lm_label_ids: Tensor,
        contrastive_labels: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids_corrupt: Tensor,
        attention_mask_corrupt: Tensor,
        img_feats: Tensor,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        mlm_result = self.mlm_forward(
            input_ids_masked,
            lm_label_ids,
            token_type_ids,
            attention_mask,
            img_feats,
            position_ids,
            head_mask,
        )

        contrastive_loss_result = self.contrastive_forward(
            input_ids_corrupt,
            token_type_ids_corrupt,
            attention_mask_corrupt,
            img_feats,
            contrastive_labels,
            position_ids,
            head_mask,
        )
        losses = {**mlm_result["losses"], **contrastive_loss_result}
        return {"losses": losses}
