# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/ChenRocks/UNITER/
# and adapted for MMF.

import collections
import logging
import random
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


class UNITERImageEmbeddings(nn.Module):
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

    @dataclass
    class Config:
        hidden_size: int = 768
        eps: float = 1e-12
        hidden_dropout_prob: float = 0
        random_init: bool = False
        bert_model_name: str = "bert-base-uncased"
        text_embeddings: Any = field(default_factory=lambda: {})
        image_embeddings: UNITERImageEmbeddings.Config = UNITERImageEmbeddings.Config()
        encoder: Any = field(default_factory=lambda: {})

    def __init__(self, config):
        super().__init__()
        self.config = config = OmegaConf.create({**asdict(self.Config()), **config})

        bert_config = BertConfig.from_pretrained(config.bert_model_name)
        bert_config.update(config.text_embeddings)
        self.text_embeddings = BertEmbeddings(bert_config)

        self.img_embeddings = UNITERImageEmbeddings(config.image_embeddings)

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
        output = self.text_embeddings(
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
        img_type_embeddings = self.text_embeddings.token_type_embeddings(img_type_ids)
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
        img_masks=None,
        txt_type_ids=None,
        img_type_ids=None,
    ):
        txt_emb = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_pos_feat, img_masks, img_type_ids
        )
        embedding_output = torch.cat([txt_emb, img_emb], dim=1)
        return embedding_output

    def forward(
        self,
        input_ids,
        position_ids,
        img_feat,
        img_pos_feat,
        attention_mask,
        img_masks=None,
        output_hidden_states=False,
        txt_type_ids=None,
        img_type_ids=None,
        input_modality="VL",
    ):
        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        # https://github.com/huggingface/transformers/issues/542 for details
        # on why we add very negative values to attention scores
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_modality == "V":
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids
            )
        elif input_modality == "L":
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


def _process_head_outputs(dataset_name, losses, sample_list, outputs):
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
    """

    @dataclass
    class Config(UNITERModelBase.Config):
        heads: Any = MISSING
        tasks: Any = MISSING

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.uniter = UNITERModelBase(self.config)

        self.heads = nn.ModuleDict()
        head_configs = self.config.get("heads", {})

        self.tasks = self.config.tasks
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(",")

        for task in self.tasks:
            head_config = head_configs[task]
            head_type = head_config.get("type", "mlp")
            head_class = registry.get_transformer_head_class(head_type)
            self.heads[task] = head_class(head_config)

        self.init_losses()

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

    def forward(self, processed_sample_list):
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

    @dataclass
    class Config(UNITERModelBase.Config):
        heads: Any = MISSING
        tasks: Any = MISSING
        mask_probability: float = 0

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.uniter = UNITERModelBase(self.config)

        self.heads = nn.ModuleDict()
        head_configs = self.config.get("heads", {})

        self.tasks = self.config.tasks
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(",")

        for task in self.tasks:
            head_config = head_configs[task]
            head_type = head_config.get("type", "mlp")
            head_class = registry.get_transformer_head_class(head_type)
            if head_type == "mrfr":
                self.heads[task] = head_class(
                    head_config, self.uniter.img_embeddings.img_linear.weight.T
                )
            else:
                self.heads[task] = head_class(head_config)

        self.init_losses()

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

    def forward(self, processed_sample_list):
        assert "is_correct" in processed_sample_list, (
            "UNITER pretraining requires mismatched captions for Image-Text-Matching."
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

    def _process_sample_list_for_pretraining(self, processed_sample_list):
        task = processed_sample_list["task"]
        if task == "mrc" or task == "mrfr":
            self._add_image_feat_masked(processed_sample_list)
            # mrc assumes cls prob is a key in sample list,
            # having cls prob as a key in sample list makes it easier
            # mask negative pairs due to mismatched captions
            processed_sample_list["cls_prob"] = torch.tensor(
                processed_sample_list["image_info_0"]["cls_prob"]
            )

        if not task == "itm" and not task == "wra":
            self._remove_mismatched_captions(processed_sample_list)

    def _add_image_feat_masked(self, processed_sample_list):
        mask_prob = self.config.get("mask_probability", 0)
        img_feat_masked = torch.clone(processed_sample_list["image_feat"])
        num_feat = img_feat_masked.size(1)

        img_masks = [
            self._get_img_mask(mask_prob, num_feat)
            for _ in range(img_feat_masked.size(0))
        ]
        img_masks = torch.tensor(img_masks).bool().to(img_feat_masked.device)
        img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat_masked)
        processed_sample_list["image_feat_masked"] = img_feat_masked.data.masked_fill(
            img_masks_ext, 0
        )
        processed_sample_list["image_mask"] = img_masks

    def _get_img_mask(self, mask_prob, num_bb):
        img_mask = [random.random() < mask_prob for _ in range(num_bb)]
        if not any(img_mask):
            # at least mask 1
            img_mask[random.choice(range(num_bb))] = True
        return img_mask

    def _preprocess_mlm(self, processed_sample_list):
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

    def _preprocess_itm(self, processed_sample_list):
        assert "is_correct" in processed_sample_list

        itm_labels = {"is_correct": processed_sample_list["is_correct"]}
        processed_sample_list["itm_labels"] = itm_labels

    def _preprocess_mrc(self, processed_sample_list):
        assert "cls_prob" in processed_sample_list
        assert "image_mask" in processed_sample_list
        assert "image_feat_masked" in processed_sample_list

        mrc_label_key = self.heads["mrc"].config.mrc_label_key
        mrc_mask_key = self.heads["mrc"].config.mrc_mask_key

        image_mask = processed_sample_list["image_mask"]
        cls_prob = processed_sample_list["cls_prob"].to(image_mask.device)
        img_masks_ext = image_mask.unsqueeze(-1).expand_as(cls_prob)  # (n, m, d)
        cls_dim = cls_prob.size(2)
        cls_prob = cls_prob[img_masks_ext].contiguous().view(-1, cls_dim)
        processed_sample_list[mrc_label_key] = cls_prob

        bs = image_mask.size(0)
        sentence_len = processed_sample_list["input_ids"].size(1)
        padding_for_txt = torch.zeros((bs, sentence_len)).to(image_mask)
        concat_mask = torch.cat([padding_for_txt, image_mask], dim=-1)
        processed_sample_list[mrc_mask_key] = concat_mask
        processed_sample_list["image_feat"] = processed_sample_list["image_feat_masked"]

    def _preprocess_mrfr(self, processed_sample_list):
        assert "image_mask" in processed_sample_list
        assert "image_feat_masked" in processed_sample_list

        mrfr_target_key = self.heads["mrfr"].config.mrfr_target_key
        mrfr_mask_key = self.heads["mrfr"].config.mrfr_mask_key

        image_mask = processed_sample_list["image_mask"]
        image_feat = processed_sample_list["image_feat"]
        img_masks_ext = image_mask.unsqueeze(-1).expand_as(image_feat)  # (n, m, d)

        feat_dim = image_feat.size(2)
        feat_targets = image_feat[img_masks_ext].contiguous().view(-1, feat_dim)
        processed_sample_list[mrfr_target_key] = feat_targets

        bs = image_mask.size(0)
        sentence_len = processed_sample_list["input_ids"].size(1)
        padding_for_txt = torch.zeros((bs, sentence_len)).to(image_mask)
        concat_mask = torch.cat([padding_for_txt, image_mask], dim=-1)
        processed_sample_list[mrfr_mask_key] = concat_mask
        processed_sample_list["image_feat"] = processed_sample_list["image_feat_masked"]

    def _preprocess_wra(self, processed_sample_list):
        assert "is_correct" in processed_sample_list

        ot_inputs_key = self.heads["wra"].config.ot_inputs_key
        wra_label_key = self.heads["wra"].config.wra_label_key

        txt_lens = [i.size(0) for i in processed_sample_list["input_ids"]]
        num_bbs = [f.size(0) for f in processed_sample_list["image_feat"]]

        def _compute_pad(lens):
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

    def _remove_mismatched_captions(self, processed_sample_list):
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
    class Config(UNITERModelBase.Config):
        heads: Any = MISSING
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
        if self.do_pretraining:
            self.uniter = UNITERForPretraining(self.config)
        else:
            self.uniter = UNITERForClassification(self.config)

        self.tasks = self.config.tasks
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(",")

    def init_losses(self):
        pass

    def add_pos_feat(self, sample_list):
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

        self.add_pos_feat(sample_list)
        return sample_list

    def forward(self, sample_list):
        sample_list = self.add_custom_params(sample_list)
        return self.uniter(sample_list)

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
