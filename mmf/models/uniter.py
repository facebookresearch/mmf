# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/ChenRocks/UNITER/
# and adapted for MMF.

import collections
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.losses import MMFLoss
from mmf.utils.general import retry_n
from omegaconf import MISSING, OmegaConf
from torch import nn
from transformers.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertPooler


logger = logging.getLogger()


class UniterImageEmbeddings(nn.Module):
    """
    Image Embeddings used by UNITER.
    Code modified from https://github.com/ChenRocks/UNITER/blob/master/model/model.py
    Performs a linear projection then normalization over image and position features.
    """

    @dataclass
    class Config:
        img_dim: int = 2048
        hidden_size: int = 768
        eps: float = 1e-12
        hidden_dropout_prob: float = 0
        pos_dim: int = 7

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        config = OmegaConf.create({**asdict(self.Config()), **config})

        self.img_linear = nn.Linear(config.img_dim, config.hidden_size)
        self.img_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.eps)
        self.pos_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.eps)
        self.pos_linear = nn.Linear(config.pos_dim, config.hidden_size)
        self.mask_embedding = nn.Embedding(2, config.img_dim, padding_idx=0)

        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, type_embeddings, img_masks=None):
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


class UniterModelBase(nn.Module):
    """ Modification for Joint Vision-Language Encoding
    """

    @dataclass
    class TextEmbeddingConfig:
        vocab_size: int = 30522
        hidden_size: int = 768
        max_position_embeddings: int = 512
        eps: float = 1e-12
        hidden_dropout_prob: float = 0
        pad_token_id: int = 0
        type_vocab_size: int = 2

    @dataclass
    class Config:
        hidden_size: int = 768
        eps: float = 1e-12
        hidden_dropout_prob: float = 0
        random_init: bool = False
        bert_model_name: str = "bert-base-uncased"
        text_embeddings: Any = field(default_factory=lambda: {})
        image_embeddings: UniterImageEmbeddings.Config = UniterImageEmbeddings.Config()
        encoder: Any = field(default_factory=lambda: {})

    def __init__(self, config):
        super().__init__()
        self.config = config = OmegaConf.create({**asdict(self.Config()), **config})

        text_embedding_config = OmegaConf.create(
            {**asdict(self.TextEmbeddingConfig()), **config.text_embeddings}
        )
        bert_config = BertConfig.from_pretrained(config.bert_model_name)
        bert_config.update(text_embedding_config)
        self.embeddings = BertEmbeddings(bert_config)

        self.img_embeddings = UniterImageEmbeddings(config.image_embeddings)

        bert_model_name = config["bert_model_name"]
        hf_config = retry_n(
            6,
            BertConfig.from_pretrained,
            bert_model_name,
            **OmegaConf.to_container(config.encoder),
        )
        hf_config.update(config.encoder)
        if config["random_init"]:
            self.encoder = BertModel(hf_config).encoder
        else:
            self.encoder = retry_n(
                6, BertModel.from_pretrained, bert_model_name, config=hf_config
            ).encoder

        self.pooler = BertPooler(config)

    def _compute_txt_embeddings(self, input_ids, position_ids, token_type_ids=None):
        output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
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
        # be ok with embeddings with padding
        # TODO: add gather_index and require less work
        # # align back to most compact input
        # gather_index = gather_index.unsqueeze(-1).expand(
        #     -1, -1, self.config.hidden_size
        # )
        # embedding_output = torch.gather(
        #     torch.cat([txt_emb, img_emb], dim=1), dim=1, index=gather_index
        # )
        embedding_output = torch.cat([txt_emb, img_emb], dim=1)
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
        output_hidden_states=False,
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
            attention_mask=extended_attention_mask,
            output_hidden_states=output_hidden_states,
        )
        if not output_hidden_states:
            encoded_layers = encoded_layers[-1]
        return encoded_layers


@registry.register_model("uniter")
class Uniter(BaseModel):
    """ Modification for Joint Vision-Language Encoding
    """

    @dataclass
    class Config(UniterModelBase.Config):
        heads: Any = MISSING
        tasks: Any = MISSING

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/uniter/defaults.yaml"

    def build(self):
        self.uniter = UniterModelBase(self.config)

        self.heads = nn.ModuleDict()
        head_configs = self.config.get("heads", {})

        self.tasks = self.config.tasks
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(",")

        for task in self.tasks:
            head_config = head_configs[task]
            head_type = head_config.get("type", "mlp")
            head_class = registry.get_transformer_head_class(head_type)
            if head_type == "wra":
                self.heads[task] = head_class(
                    head_config, self.uniter.img_embeddings.weight
                )
            else:
                self.heads[task] = head_class(head_config)

    def init_losses(self):
        self.losses = nn.ModuleDict()
        loss_configs = self.config.get("losses", {})
        for task in self.tasks:
            if task not in loss_configs:
                logger.warning(
                    f"No loss defined for {task}. Head is expected "
                    + "to return dict with 'losses'"
                )
                continue
            loss_config = loss_configs[task]
            self.losses[task] = MMFLoss(loss_config)

    def add_pos_feat(self, sample_list):
        assert "image_info_0" in sample_list
        assert "bbox" in sample_list["image_info_0"]

        bboxs = torch.tensor(sample_list["image_info_0"]["bbox"])  # (bs, num_feats, 4)
        img_h = (
            torch.tensor(sample_list["image_info_0"]["image_height"])
            .unsqueeze(1)
            .unsqueeze(1)
        )  # (bs,)
        img_w = (
            torch.tensor(sample_list["image_info_0"]["image_width"])
            .unsqueeze(1)
            .unsqueeze(1)
        )  # (bs,)

        norm_xy = torch.clone(bboxs)
        max_image_size = torch.cat([img_w, img_h, img_w, img_h], dim=-1)
        norm_xy /= max_image_size

        num_feat = bboxs.size(1)
        expanded_img_w = img_w.expand(-1, num_feat, -1)
        expanded_img_h = img_h.expand(-1, num_feat, -1)
        area = expanded_img_w * expanded_img_h

        pos_feat = torch.cat(
            [norm_xy, expanded_img_w, expanded_img_h, area], dim=-1
        ).to(sample_list["image_feature_0"])
        sample_list["img_pos_feat"] = pos_feat

    def add_custom_params(self, sample_list):
        image_feat = sample_list["image_feat"] = sample_list["image_feature_0"]

        image_info = getattr(sample_list, "image_info_0", {})
        image_dim = getattr(image_info, "max_features", None)
        sample_list["image_dim"] = image_dim

        image_mask = torch.arange(image_feat.size(-2), device=image_feat.device).expand(
            image_feat.size()[:-1]
        )
        if len(image_dim.size()) < len(image_mask.size()):
            image_dim = image_dim.unsqueeze(-1)
            assert len(image_dim.size()) == len(image_mask.size())
        image_mask = image_mask < image_dim
        sample_list["image_mask"] = image_mask.long()

        attention_mask = torch.cat(
            (sample_list["input_mask"], sample_list["image_mask"]), dim=-1
        )
        sample_list["attention_mask"] = attention_mask
        task_index = torch.randint(len(self.tasks), (1,)).item()
        sample_list["task"] = self.tasks[task_index]
        sample_list["position_ids"] = torch.arange(
            0,
            sample_list["input_ids"].size(1),
            dtype=torch.long,
            device=image_feat.device,
        ).unsqueeze(0)
        sample_list["gather_index"] = None

        self.add_pos_feat(sample_list)
        return sample_list

    def forward(self, sample_list):
        sample_list = self.add_custom_params(sample_list)

        task = sample_list["task"]
        if task == "mlm" or task == "itm":
            img_masks = None
        else:
            img_masks = sample_list["image_mask"]

        sequence_output = self.uniter(
            sample_list["input_ids"],
            sample_list["position_ids"],
            sample_list["image_feat"],
            sample_list["img_pos_feat"],
            sample_list["attention_mask"],
            sample_list["gather_index"],
            output_hidden_states=False,
            img_masks=img_masks,
        )

        outputs = self.heads[task](sequence_output, processed_sample_list=sample_list)

        if isinstance(outputs, collections.MutableMapping) and "losses" in outputs:
            return outputs

        logits = outputs
        if isinstance(outputs, collections.MutableMapping) and "scores" in outputs:
            logits = outputs["scores"]
        logits = logits.contiguous().view(-1, logits.size(-1))
        output = self.losses[sample_list.dataset_name](sample_list, {"scores": logits})
        return {"losses": output, "scores": logits}

    def get_attention_mask(self, sample_list, text_embedding, image_embedding):
        image_mask = getattr(sample_list, "image_mask", None)

        if image_mask is not None and sample_list.input_mask is not None:
            attention_mask = torch.cat((sample_list.input_mask, image_mask), dim=-1)
        elif image_mask is not None:
            text_mask = torch.ones(
                text_embedding.size()[:-1],
                dtype=text_embedding.dtype,
                device=text_embedding.device,
            )
            attention_mask = torch.cat((image_mask, text_mask), dim=-1)
        elif sample_list.input_mask is not None:
            image_mask = torch.ones(
                image_embedding.size()[:-1],
                dtype=image_embedding.dtype,
                device=image_embedding.device,
            )
            attention_mask = torch.cat((image_mask, sample_list.input_mask), dim=-1)
        else:
            attention_mask = None

        return attention_mask
