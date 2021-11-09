# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/ChenRocks/UNITER/
# and adapted for MMF.

import collections
import copy
import logging
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.losses import MMFLoss
from mmf.utils.general import retry_n
from omegaconf import MISSING, DictConfig, OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import BertConfig, BertEmbeddings, BertModel


NUM_RETRIES = 6
EMPTY_CONFIG = OmegaConf.create({})
DEFAULT_PRETRAINING_HEAD_CONFIGS = {
    "mlm": {"type": "mlm"},
    "itm": {"type": "itm"},
    "mrc": {"type": "mrc"},
    "mrfr": {"type": "mrfr"},
    "wra": {"type": "wra"},
}
DEFAULT_PRETRAINING_TASKS = "mlm,itm,mrc,mrfr,wra"


logger = logging.getLogger()


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


def _process_head_outputs(
    dataset_name: str,
    losses: Dict[str, Any],
    sample_list: Dict[str, Tensor],
    outputs: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    if isinstance(outputs, collections.MutableMapping) and "losses" in outputs:
        return outputs

    logits = outputs
    if isinstance(outputs, collections.MutableMapping) and "scores" in outputs:
        logits = outputs["scores"]
    logits = logits.contiguous().view(-1, logits.size(-1))
    output = losses[dataset_name](sample_list, {"scores": logits})
    return {"losses": output, "scores": logits}


class UNITERForClassification(nn.Module):
    """ UNITER wrapper for classification

    Example params:
        head_configs = {"vqa2": {"type": "mlp", "num_labels": 3129}}
        losses_configs = {"vqa2": "logit_bce"}
        tasks = "vqa2"
    """

    def __init__(
        self,
        head_configs: Dict,
        loss_configs: Dict,
        tasks: Union[str, List],
        random_init: bool = False,
        bert_model_name: str = "bert-base-uncased",
        img_dim: int = 2048,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0,
        text_embeddings: Any = EMPTY_CONFIG,
        encoder: Any = EMPTY_CONFIG,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.loss_configs = loss_configs
        self.uniter = UNITERModelBase(
            random_init=random_init,
            bert_model_name=bert_model_name,
            img_dim=img_dim,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            text_embeddings=text_embeddings,
            encoder=encoder,
        )

        self.heads = nn.ModuleDict()
        self.tasks = tasks
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(",")

        for task in self.tasks:
            assert task in head_configs, (
                f"Task {task} is specified in your model configs"
                + " but there is no head configured for the task."
                + "Head configs can be added under model_config.heads"
                + "in your yaml configs. Either remove this task if UNITER"
                + " is not meant to run on a dataset named {task}"
                + " or add a head config."
            )
            head_config = head_configs[task]
            head_type = head_config.get("type", "mlp")
            head_class = registry.get_transformer_head_class(head_type)
            self.heads[task] = head_class(head_config)

        self.init_losses()

    def init_losses(self):
        self.losses = nn.ModuleDict()
        for task in self.tasks:
            if task not in self.loss_configs:
                logger.warning(
                    f"No loss defined for {task}. Head is expected "
                    + "to return dict with 'losses'"
                )
                continue
            loss_config = self.loss_configs[task]
            self.losses[task] = MMFLoss(loss_config)

    def forward(self, processed_sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sequence_output = self.uniter(
            processed_sample_list["input_ids"],
            processed_sample_list["position_ids"],
            processed_sample_list["image_feat"],
            processed_sample_list["img_pos_feat"],
            processed_sample_list["attention_mask"],
            img_masks=processed_sample_list["image_mask"],
            output_hidden_states=False,
        )
        dataset_name = processed_sample_list["dataset_name"]
        outputs = self.heads[dataset_name](
            sequence_output, processed_sample_list=processed_sample_list
        )

        return _process_head_outputs(
            dataset_name, self.losses, processed_sample_list, outputs
        )


class UNITERForPretraining(nn.Module):
    """ UNITER wrapper for pretraining
    """

    def __init__(
        self,
        head_configs: Optional[Dict] = None,
        loss_configs: Optional[Dict] = None,
        tasks: Union[List, str] = DEFAULT_PRETRAINING_TASKS,
        mask_probability: float = 0,
        random_init: bool = False,
        bert_model_name: str = "bert-base-uncased",
        img_dim: int = 2048,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0,
        text_embeddings: Any = EMPTY_CONFIG,
        encoder: Any = EMPTY_CONFIG,
        *args,
        **kwargs,
    ):
        super().__init__()
        if head_configs is None:
            head_configs = copy.deepcopy(DEFAULT_PRETRAINING_HEAD_CONFIGS)

        if loss_configs is None:
            loss_configs = {}

        self.loss_configs = loss_configs
        self.mask_probability = mask_probability
        self.uniter = UNITERModelBase(
            random_init=random_init,
            bert_model_name=bert_model_name,
            img_dim=img_dim,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            text_embeddings=text_embeddings,
            encoder=encoder,
        )

        self.heads = nn.ModuleDict()

        self.tasks = tasks
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(",")

        for task in self.tasks:
            head_config = head_configs[task]
            head_type = head_config.get("type", "mlp")
            head_class = registry.get_transformer_head_class(head_type)
            if head_type == "mrfr":
                self.heads[task] = head_class(
                    self.uniter.img_embeddings.img_linear.weight, **head_config
                )
            elif head_type in ("itm", "mlm", "mlp"):
                self.heads[task] = head_class(head_config)
            else:
                self.heads[task] = head_class(**head_config)

        self.init_losses()

    def init_losses(self):
        self.losses = nn.ModuleDict()
        for task in self.tasks:
            if task not in self.loss_configs:
                logger.warning(
                    f"No loss defined for {task}. Head is expected "
                    + "to return dict with 'losses'"
                )
                continue
            loss_config = self.loss_configs[task]
            self.losses[task] = MMFLoss(loss_config)

    def forward(self, processed_sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert "is_correct" in processed_sample_list, (
            "UNITER pretraining requires mismatched captions."
            + " Please add 'false_caption': true under dataset_config in your "
            + "yaml configs."
        )

        self._process_sample_list_for_pretraining(processed_sample_list)

        task = processed_sample_list["task"]
        if task == "mlm":
            self._preprocess_mlm(processed_sample_list)
        elif task == "itm":
            self._preprocess_itm(processed_sample_list)
        elif task == "mrc":
            self._preprocess_mrc(processed_sample_list)
        elif task == "mrfr":
            self._preprocess_mrfr(processed_sample_list)
        elif task == "wra":
            self._preprocess_wra(processed_sample_list)
        else:
            raise ValueError(f"Task {task} is not supported for pretraining!")

        sequence_output = self.uniter(
            processed_sample_list["input_ids"],
            processed_sample_list["position_ids"],
            processed_sample_list["image_feat"],
            processed_sample_list["img_pos_feat"],
            processed_sample_list["attention_mask"],
            img_masks=processed_sample_list["image_mask"],
            output_hidden_states=False,
        )
        dataset_name = processed_sample_list["dataset_name"]
        outputs = self.heads[task](
            sequence_output, processed_sample_list=processed_sample_list
        )

        return _process_head_outputs(
            dataset_name, self.losses, processed_sample_list, outputs
        )

    def _process_sample_list_for_pretraining(
        self, processed_sample_list: Dict[str, Tensor]
    ):
        task = processed_sample_list["task"]
        if task in ("mrfr", "mrc"):
            self._add_image_feat_masked(processed_sample_list)
            # mrc assumes cls prob is a key in sample list,
            # having cls prob as a key in sample list makes it easier
            # mask negative pairs due to mismatched captions
            processed_sample_list["cls_prob"] = torch.tensor(
                processed_sample_list["image_info_0"]["cls_prob"]
            )

        if task not in ("wra", "itm"):
            self._remove_mismatched_captions(processed_sample_list)

    def _add_image_feat_masked(self, processed_sample_list: Dict[str, Tensor]):
        img_feat_masked = torch.clone(processed_sample_list["image_feat"])
        num_feat = img_feat_masked.size(1)

        img_masks = [
            self._get_img_mask(self.mask_probability, num_feat)
            for _ in range(img_feat_masked.size(0))
        ]
        img_masks = torch.tensor(img_masks).bool().to(img_feat_masked.device)
        img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat_masked)
        processed_sample_list["image_feat_masked"] = img_feat_masked.data.masked_fill(
            img_masks_ext, 0
        )
        processed_sample_list["image_mask"] = img_masks

    def _get_img_mask(self, mask_prob: float, num_bb: int) -> Tensor:
        img_mask = list(map(bool, np.random.binomial(1, mask_prob, num_bb)))
        if not any(img_mask):
            # at least mask 1
            img_mask[random.choice(range(num_bb))] = True
        return img_mask

    def _preprocess_mlm(self, processed_sample_list: Dict[str, Tensor]):
        assert "lm_label_ids" in processed_sample_list
        assert "input_ids_masked" in processed_sample_list

        ignore_index = self.heads["mlm"].config.ignore_index
        mlm_labels = {}
        mlm_labels["text"] = processed_sample_list["lm_label_ids"]
        mlm_labels["image"] = torch.full(
            processed_sample_list["image_feat"].shape[:2],
            fill_value=ignore_index,
            dtype=torch.long,
            device=mlm_labels["text"].device,
        )
        mlm_labels["combined_labels"] = torch.cat(
            [mlm_labels["text"], mlm_labels["image"]], dim=-1
        )
        processed_sample_list["mlm_labels"] = mlm_labels
        processed_sample_list["input_ids"] = processed_sample_list["input_ids_masked"]

    def _preprocess_itm(self, processed_sample_list: Dict[str, Tensor]):
        assert "is_correct" in processed_sample_list

        processed_sample_list["itm_labels"] = {
            "is_correct": processed_sample_list["is_correct"]
        }

    def _get_feature_mask(self, image_mask, sentence_len):
        bs = image_mask.size(0)
        padding_for_txt = torch.zeros((bs, sentence_len)).to(image_mask)
        concat_mask = torch.cat([padding_for_txt, image_mask], dim=-1)
        return concat_mask

    def _preprocess_mrc(self, processed_sample_list: Dict[str, Tensor]):
        assert "cls_prob" in processed_sample_list
        assert "image_mask" in processed_sample_list
        assert "image_feat_masked" in processed_sample_list

        mrc_label_key = self.heads["mrc"].mrc_label_key
        mrc_mask_key = self.heads["mrc"].mrc_mask_key

        image_mask = processed_sample_list["image_mask"]
        cls_prob = processed_sample_list["cls_prob"].to(image_mask.device)
        img_masks_ext = image_mask.unsqueeze(-1).expand_as(cls_prob)  # (n, m, d)
        cls_dim = cls_prob.size(2)
        cls_prob = cls_prob[img_masks_ext].contiguous().view(-1, cls_dim)
        processed_sample_list[mrc_label_key] = cls_prob

        sentence_len = processed_sample_list["input_ids"].size(1)
        processed_sample_list[mrc_mask_key] = self._get_feature_mask(
            image_mask, sentence_len
        )
        processed_sample_list["image_feat"] = processed_sample_list["image_feat_masked"]

    def _preprocess_mrfr(self, processed_sample_list: Dict[str, Tensor]):
        assert "image_mask" in processed_sample_list
        assert "image_feat_masked" in processed_sample_list

        mrfr_target_key = self.heads["mrfr"].mrfr_target_key
        mrfr_mask_key = self.heads["mrfr"].mrfr_mask_key

        image_mask = processed_sample_list["image_mask"]
        image_feat = processed_sample_list["image_feat"]
        img_masks_ext = image_mask.unsqueeze(-1).expand_as(image_feat)  # (n, m, d)

        feat_dim = image_feat.size(2)
        feat_targets = image_feat[img_masks_ext].contiguous().view(-1, feat_dim)
        processed_sample_list[mrfr_target_key] = feat_targets

        sentence_len = processed_sample_list["input_ids"].size(1)
        processed_sample_list[mrfr_mask_key] = self._get_feature_mask(
            image_mask, sentence_len
        )
        processed_sample_list["image_feat"] = processed_sample_list["image_feat_masked"]

    def _preprocess_wra(self, processed_sample_list: Dict[str, Tensor]):
        assert "is_correct" in processed_sample_list

        ot_inputs_key = self.heads["wra"].ot_inputs_key
        wra_label_key = self.heads["wra"].wra_label_key

        txt_lens = [i.size(0) for i in processed_sample_list["input_ids"]]
        num_bbs = [f.size(0) for f in processed_sample_list["image_feat"]]

        def _compute_pad(lens: List[int]):
            max_len = max(lens)
            pad = torch.zeros(len(lens), max_len)
            for i, l in enumerate(lens):
                pad.data[i, l:].fill_(1)
            return pad

        device = processed_sample_list["input_ids"].device
        txt_pad = _compute_pad(txt_lens).to(device).bool()
        img_pad = _compute_pad(num_bbs).to(device).bool()

        ot_inputs = {"txt_pad": txt_pad, "img_pad": img_pad}

        processed_sample_list[ot_inputs_key] = ot_inputs
        processed_sample_list[wra_label_key] = processed_sample_list["is_correct"]

    def _remove_mismatched_captions(self, processed_sample_list: Dict[str, Tensor]):
        assert "is_correct" in processed_sample_list

        pos_pairs = processed_sample_list["is_correct"].ne(0)
        pos_pairs_mask = torch.where(pos_pairs.any(), pos_pairs, pos_pairs.new([True]))
        tensor_names = [
            "input_ids",
            "input_mask",
            "image_feat",
            "img_pos_feat",
            "attention_mask",
            "image_mask",
            "image_feat_masked",
            "lm_label_ids",
            "cls_prob",
        ]
        for name in tensor_names:
            x = processed_sample_list.get(name)
            if x is None:
                continue
            if x.dim() == 1:
                assert x.size(0) == pos_pairs_mask.size(0), (
                    f"tensor {name} has shape {x.shape} but expected "
                    + f"{pos_pairs_mask.size(0)} at dim 0."
                )
                x = x[pos_pairs_mask]
            else:
                x = x[pos_pairs_mask, ::]


@registry.register_model("uniter")
class UNITER(BaseModel):
    """ Modification for Joint Vision-Language Encoding
    """

    @dataclass
    class Config:
        random_init: bool = False
        bert_model_name: str = "bert-base-uncased"
        img_dim: int = 2048
        hidden_size: int = 768
        hidden_dropout_prob: float = 0
        text_embeddings: Any = field(default_factory=lambda: {})
        encoder: Any = field(default_factory=lambda: {})
        heads: Any = MISSING
        losses: Any = field(default_factory=lambda: {})
        tasks: Any = MISSING
        do_pretraining: bool = False

    def __init__(self, config):
        super().__init__(config)
        self.config = OmegaConf.create({**asdict(self.Config()), **config})
        self.do_pretraining = self.config.do_pretraining

    @classmethod
    def config_path(cls):
        return "configs/models/uniter/defaults.yaml"

    def build(self):
        params = dict(
            **self.config,
            head_configs=self.config.heads,
            loss_configs=self.config.losses,
        )
        if self.do_pretraining:
            self.uniter = UNITERForPretraining(**params)
        else:
            self.uniter = UNITERForClassification(**params)

        self.tasks = self.config.tasks
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(",")

    def init_losses(self):
        """
        Defer loss management to submodels,
        do nothing when called by build_model.
        """

    def add_pos_feat(self, sample_list: Dict[str, Tensor]):
        assert "image_info_0" in sample_list
        assert "bbox" in sample_list["image_info_0"]

        # (x1, y1, x2, y2), dim = (bs, num_feats, 4)
        bboxs = torch.tensor(sample_list["image_info_0"]["bbox"])[:, :, :4]
        norm_xy = torch.clone(bboxs)
        # if bboxs are not normalized, just do it here
        if norm_xy[0, 0, 0] < 1:
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
            max_image_size = torch.cat([img_w, img_h, img_w, img_h], dim=-1)
            max_image_size = max_image_size.to(norm_xy.device)
            norm_xy /= max_image_size

        bbox_w = (norm_xy[:, :, 2] - norm_xy[:, :, 0]).unsqueeze(-1)
        bbox_h = (norm_xy[:, :, 3] - norm_xy[:, :, 1]).unsqueeze(-1)
        area = bbox_w * bbox_h
        # normalized (x1, y1, x2, y2, w, h, area)
        pos_feat = torch.cat([norm_xy, bbox_w, bbox_h, area], dim=-1).to(
            sample_list["image_feature_0"]
        )
        sample_list["img_pos_feat"] = pos_feat

    def add_custom_params(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
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

        self.add_pos_feat(sample_list)
        return sample_list

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_list = self.add_custom_params(sample_list)
        return self.uniter(sample_list)

    def get_attention_mask(
        self,
        sample_list: Dict[str, Tensor],
        text_embedding: Tensor,
        image_embedding: Tensor,
    ) -> Tensor:
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
