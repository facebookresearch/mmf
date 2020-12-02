# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import logging
import math

import torch
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer
from mmf.utils.build import build_image_encoder
from omegaconf import OmegaConf
from torch import nn
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
)


logger = logging.getLogger(__name__)


@registry.register_model("m4c")
class M4C(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.mmt_config = BertConfig(**self.config.mmt)
        self._datasets = registry.get("config").datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/m4c/defaults.yaml"

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_mmt()
        self._build_output()

    def _build_encoder_config(self):
        return OmegaConf.create(
            {
                "type": "finetune_faster_rcnn_fpn_fc7",
                "params": {
                    "in_dim": 2048,
                    "weights_file": "models/detectron.defaults/fc7_w.pkl",
                    "bias_file": "models/detectron.defaults/fc7_b.pkl",
                    "model_data_dir": self.config.model_data_dir,
                },
            }
        )

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768

        self.text_bert_config = BertConfig(**self.config.text_bert)
        if self.config.text_bert_init_from_bert_base:
            self.text_bert = TextBert.from_pretrained(
                "bert-base-uncased", config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append(
                {"module": self.text_bert, "lr_scale": self.config.lr_scale_text_bert}
            )
        else:
            logger.info("NOT initializing text_bert from BERT_BASE")
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            logger.info(
                f"Projecting text_bert output to {self.mmt_config.hidden_size} dim"
            )

            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def _build_obj_encoding(self):
        # object appearance feature: Faster R-CNN
        self.obj_faster_rcnn_fc7 = build_image_encoder(
            self._build_encoder_config(), direct_features=True
        )
        # apply smaller lr to pretrained Faster R-CNN fc7
        self.finetune_modules.append(
            {"module": self.obj_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        )
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.obj_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.obj_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)

    def _build_ocr_encoding(self):
        self.remove_ocr_fasttext = getattr(
            self.config.ocr, "remove_ocr_fasttext", False
        )
        self.remove_ocr_phoc = getattr(self.config.ocr, "remove_ocr_phoc", False)
        self.remove_ocr_frcn = getattr(self.config.ocr, "remove_ocr_frcn", False)
        self.remove_ocr_semantics = getattr(
            self.config.ocr, "remove_ocr_semantics", False
        )
        self.remove_ocr_bbox = getattr(self.config.ocr, "remove_ocr_bbox", False)

        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = build_image_encoder(
            self._build_encoder_config(), direct_features=True
        )
        self.finetune_modules.append(
            {"module": self.ocr_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        )

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.ocr.mmt_in_dim, self.mmt_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.ocr_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config)

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append(
            {"module": self.mmt, "lr_scale": self.config.lr_scale_mmt}
        )

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)

        # fixed answer vocabulary scores
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        # remove the OCR copying dimensions in LoRRA's classifier output
        # (OCR copying will be handled separately)
        num_choices -= self.config.classifier.ocr_max_num
        self.classifier = ClassifierLayer(
            self.config.classifier.type,
            in_dim=self.mmt_config.hidden_size,
            out_dim=num_choices,
            **self.config.classifier.params,
        )

        self.answer_processor = registry.get(self._datasets[0] + "_answer_processor")

    def forward(self, sample_list):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(sample_list, fwd_results)
        self._forward_obj_encoding(sample_list, fwd_results)
        self._forward_ocr_encoding(sample_list, fwd_results)
        self._forward_mmt_and_output(sample_list, fwd_results)

        # only keep scores in the forward pass results
        results = {"scores": fwd_results["scores"]}
        return results

    def _forward_txt_encoding(self, sample_list, fwd_results):
        fwd_results["txt_inds"] = sample_list.text

        # binary mask of valid text (question words) vs padding
        fwd_results["txt_mask"] = _get_mask(
            sample_list.text_len, sample_list.text.size(1)
        )

    def _forward_obj_encoding(self, sample_list, fwd_results):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = sample_list.image_feature_0
        obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)

        obj_feat = obj_fc7
        obj_bbox = sample_list.obj_bbox_coordinates
        obj_mmt_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        ) + self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(obj_bbox))
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results["obj_mmt_in"] = obj_mmt_in

        # binary mask of valid object vs padding
        obj_nums = sample_list.image_info_0.max_features
        fwd_results["obj_mask"] = _get_mask(obj_nums, obj_mmt_in.size(1))

    def _forward_ocr_encoding(self, sample_list, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = sample_list.context_feature_0
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = sample_list.image_feature_1[:, : ocr_fasttext.size(1), :]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        # OCR order vectors (legacy from LoRRA model; set to all zeros)
        # TODO remove OCR order vectors; they are not needed
        ocr_order_vectors = torch.zeros_like(sample_list.order_vectors)

        if self.remove_ocr_fasttext:
            ocr_fasttext = torch.zeros_like(ocr_fasttext)
        if self.remove_ocr_phoc:
            ocr_phoc = torch.zeros_like(ocr_phoc)
        if self.remove_ocr_frcn:
            ocr_fc7 = torch.zeros_like(ocr_fc7)
        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_fc7, ocr_order_vectors], dim=-1
        )
        ocr_bbox = sample_list.ocr_bbox_coordinates
        if self.remove_ocr_semantics:
            ocr_feat = torch.zeros_like(ocr_feat)
        if self.remove_ocr_bbox:
            ocr_bbox = torch.zeros_like(ocr_bbox)
        ocr_mmt_in = self.ocr_feat_layer_norm(
            self.linear_ocr_feat_to_mmt_in(ocr_feat)
        ) + self.ocr_bbox_layer_norm(self.linear_ocr_bbox_to_mmt_in(ocr_bbox))
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results["ocr_mmt_in"] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        ocr_nums = sample_list.context_info_0.max_features
        fwd_results["ocr_mask"] = _get_mask(ocr_nums, ocr_mmt_in.size(1))

    def _forward_mmt(self, sample_list, fwd_results):
        # first forward the text BERT layers
        text_bert_out = self.text_bert(
            txt_inds=fwd_results["txt_inds"], txt_mask=fwd_results["txt_mask"]
        )
        fwd_results["txt_emb"] = self.text_bert_out_linear(text_bert_out)

        mmt_results = self.mmt(
            txt_emb=fwd_results["txt_emb"],
            txt_mask=fwd_results["txt_mask"],
            obj_emb=fwd_results["obj_mmt_in"],
            obj_mask=fwd_results["obj_mask"],
            ocr_emb=fwd_results["ocr_mmt_in"],
            ocr_mask=fwd_results["ocr_mask"],
            fixed_ans_emb=self.classifier.module.weight,
            prev_inds=fwd_results["prev_inds"],
        )
        fwd_results.update(mmt_results)

    def _forward_output(self, sample_list, fwd_results):
        mmt_dec_output = fwd_results["mmt_dec_output"]
        mmt_ocr_output = fwd_results["mmt_ocr_output"]
        ocr_mask = fwd_results["ocr_mask"]

        fixed_scores = self.classifier(mmt_dec_output)
        dynamic_ocr_scores = self.ocr_ptr_net(mmt_dec_output, mmt_ocr_output, ocr_mask)
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
        fwd_results["scores"] = scores

    def _forward_mmt_and_output(self, sample_list, fwd_results):
        if self.training:
            fwd_results["prev_inds"] = sample_list.train_prev_inds.clone()
            self._forward_mmt(sample_list, fwd_results)
            self._forward_output(sample_list, fwd_results)
        else:
            dec_step_num = sample_list.train_prev_inds.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            fwd_results["prev_inds"] = torch.zeros_like(sample_list.train_prev_inds)
            fwd_results["prev_inds"][:, 0] = self.answer_processor.BOS_IDX

            # greedy decoding at test time
            for _ in range(dec_step_num):
                self._forward_mmt(sample_list, fwd_results)
                self._forward_output(sample_list, fwd_results)

                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = fwd_results["scores"].argmax(dim=-1)
                fwd_results["prev_inds"][:, 1:] = argmax_inds[:, :-1]

    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append(
                {
                    "params": list(m["module"].parameters()),
                    "lr": base_lr * m["lr_scale"],
                }
            )
            finetune_params_set.update(list(m["module"].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups

    @classmethod
    def update_registry_for_pretrained(cls, config, checkpoint, full_output):
        from omegaconf import OmegaConf

        # Hack datasets using OmegaConf
        datasets = full_output["full_config"].datasets
        dataset = datasets.split(",")[0]
        config_mock = OmegaConf.create({"datasets": datasets})
        registry.register("config", config_mock)
        registry.register(
            f"{dataset}_num_final_outputs",
            # Need to add as it is subtracted
            checkpoint["classifier.module.weight"].size(0)
            + config.classifier.ocr_max_num,
        )
        # Fix this later, when processor pipeline is available
        answer_processor = OmegaConf.create({"BOS_IDX": 1})
        registry.register(f"{dataset}_answer_processor", answer_processor)


class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output


class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(
        self,
        txt_emb,
        txt_mask,
        obj_emb,
        obj_mask,
        ocr_emb,
        ocr_mask,
        fixed_ans_emb,
        prev_inds,
    ):

        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        dec_mask = torch.zeros(
            dec_emb.size(0), dec_emb.size(1), dtype=torch.float32, device=dec_emb.device
        )
        encoder_inputs = torch.cat([txt_emb, obj_emb, ocr_emb, dec_emb], dim=1)
        attention_mask = torch.cat([txt_mask, obj_mask, ocr_mask, dec_mask], dim=1)

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        ocr_begin = txt_max_num + obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        # decoding step elements can attend to themselves in a causal manner
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = _get_causal_mask(
            dec_max_num, encoder_inputs.device
        )

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        results = {
            "mmt_seq_output": mmt_seq_output,
            "mmt_txt_output": mmt_txt_output,
            "mmt_ocr_output": mmt_ocr_output,
            "mmt_dec_output": mmt_dec_output,
        }
        return results


class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        assert extended_attention_mask.dim() == 2
        extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores


class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        # Add position and type embedding for previous predictions
        position_ids = torch.arange(seq_length, dtype=torch.long, device=ocr_emb.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings

        return dec_emb


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[i, j] = 1.0
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size * length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results
