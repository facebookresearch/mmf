# Copyright (c) Facebook, Inc. and its affiliates.

# MMBTModel, ModalEmbeddings is copied from [1]
# as we have internal dependency on transformers v2.3.
# These will be removed when we upgrade to package v2.5+.
# [1]: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_mmbt.py # noqa

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.models.interfaces.mmbt import MMBTGridHMInterface
from mmf.modules.encoders import (
    EncoderFactory,
    ImageEncoderFactory,
    ImageEncoderTypes,
    MultiModalEncoderBase,
    ResNet152ImageEncoder,
    TextEncoderFactory,
    TextEncoderTypes,
    TransformerEncoder,
)
from mmf.modules.hf_layers import replace_with_jit
from mmf.utils.checkpoint import load_pretrained_model
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from omegaconf import II, DictConfig, OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import BertForPreTraining, BertPredictionHeadTransform


# TODO: Remove after transformers package upgrade to 2.5
class MMBTConfig:
    """Configuration class to store the configuration of a `MMBT Model`.
    Args:
        config (:obj:`~transformers.PreTrainedConfig`):
            Config of the underlying Transformer models. Its values are
            copied over to use a single config.
        num_labels (:obj:`int` or :obj:`None`, optional, defaults to `None`):
            Size of final Linear layer for classification.
        modal_hidden_size (:obj:`int`, optional, defautls to 2048):
            Embedding dimension of the non-text modality encoder.
    """

    def __init__(self, config, num_labels=None, modal_hidden_size=2048):
        self.__dict__ = config.__dict__
        self.modal_hidden_size = modal_hidden_size
        if num_labels:
            self.num_labels = num_labels


# TODO: Remove after transformers package upgrade to 2.5
class ModalEmbeddings(nn.Module):
    """Generic Modal Embeddings which takes in an encoder, and a transformer embedding.
    """

    def __init__(self, config, encoder, embeddings):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.proj_embeddings = nn.Linear(config.modal_hidden_size, config.hidden_size)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(
        self,
        input_modal: Tensor,
        start_token: Optional[Tensor] = None,
        end_token: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ):
        token_embeddings = self.proj_embeddings(self.encoder(input_modal))
        seq_length = token_embeddings.size(1)

        if start_token is not None:
            start_token_embeds = self.word_embeddings(start_token)
            seq_length += 1
            token_embeddings = torch.cat(
                [start_token_embeds.unsqueeze(1), token_embeddings], dim=1
            )

        if end_token is not None:
            end_token_embeds = self.word_embeddings(end_token)
            seq_length += 1
            token_embeddings = torch.cat(
                [token_embeddings, end_token_embeds.unsqueeze(1)], dim=1
            )

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_modal.device
            )
            position_ids = position_ids.unsqueeze(0).expand(
                input_modal.size(0), seq_length
            )

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (input_modal.size(0), seq_length),
                dtype=torch.long,
                device=input_modal.device,
            )

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# TODO: Remove after transformers package upgrade to 2.5
class MMBTModel(nn.Module):
    r"""
        Outputs: `Tuple` comprising various elements depending on the configuration
            (config) and inputs:
            **last_hidden_state**: ``torch.FloatTensor`` of shape
                ``(batch_size, sequence_length, hidden_size)``. Sequence of
                hidden-states at the output of the last layer of the model.
            **pooler_output**: ``torch.FloatTensor`` of shape
                ``(batch_size, hidden_size)``. Last layer hidden-state of the
                first token of the sequence (classification token) further processed
                by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction
                (classification) objective during Bert pretraining. This output
                is usually *not* a good summary of the semantic content of the
                input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            **hidden_states**: (`optional`, returned when
                ``config.output_hidden_states=True``)
                list of ``torch.FloatTensor`` (one for the output of each layer +
                the output of the embeddings)
                of shape ``(batch_size, sequence_length, hidden_size)``:
                Hidden-states of the model at the output of each layer plus the
                initial embedding outputs.
            **attentions**: (`optional`, returned when
                ``config.output_attentions=True``) list of ``torch.FloatTensor``
                (one for each layer) of shape ``(batch_size, num_heads,
                sequence_length, sequence_length)``: Attentions weights after
                the attention softmax, used to compute the weighted average in the
                self-attention heads.
        Examples::
            # For example purposes. Not runnable.
            transformer = BertModel.from_pretrained('bert-base-uncased')
            encoder = ImageEncoder(args)
            mmbt = MMBTModel(config, transformer, encoder)
        """

    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.num_hidden_layers = config.num_hidden_layers
        self.transformer = transformer
        self.modal_encoder = ModalEmbeddings(config, encoder, transformer.embeddings)

    def forward(
        self,
        input_modal: Tensor,
        input_ids: Tensor,
        modal_start_tokens: Optional[Tensor] = None,
        modal_end_tokens: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        modal_token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        modal_position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_txt_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_txt_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = inputs_embeds.device if inputs_embeds is not None else input_ids.device

        modal_embeddings = self.modal_encoder(
            input_modal,
            start_token=modal_start_tokens,
            end_token=modal_end_tokens,
            position_ids=modal_position_ids,
            token_type_ids=modal_token_type_ids,
        )

        input_modal_shape = modal_embeddings.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.ones(
                input_txt_shape, dtype=torch.long, device=device
            )

        txt_embeddings = self.transformer.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        embedding_output = torch.cat([modal_embeddings, txt_embeddings], 1)

        input_shape = embedding_output.size()[:-1]

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        else:
            attention_mask = torch.cat(
                [
                    torch.ones(input_modal_shape, device=device, dtype=torch.long),
                    attention_mask,
                ],
                dim=1,
            )

        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        else:
            encoder_attention_mask = torch.cat(
                [torch.ones(input_modal_shape, device=device), encoder_attention_mask],
                dim=1,
            )

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the
        #   padding mask
        # - if the model is an encoder, make the mask broadcastable to
        # [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # Python builtin next is currently not supported in Torchscript
        if not torch.jit.is_scripting():
            attention_mask = attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to
        # [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_attention_mask = encoder_attention_mask[:, None, None, :]

        # Python builtin next is currently not supported in Torchscript
        if not torch.jit.is_scripting():
            encoder_attention_mask = encoder_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility

        encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0

        encoder_outputs = self.transformer.encoder(
            embedding_output,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.transformer.pooler(sequence_output)

        outputs = (
            sequence_output,
            pooled_output,
            encoder_outputs[1:],
        )  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class MMBTBase(MultiModalEncoderBase):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # Replace transformer layers with scriptable JIT layers
        replace_with_jit()

    def build(self):
        encoders = self._build_encoders(self.config)
        text_encoder, modal_encoder = encoders[0], encoders[1]
        self._encoder_config = text_encoder.config

        self._mmbt_config = MMBTConfig(
            self._encoder_config,
            num_labels=self.config.num_labels,
            modal_hidden_size=self.config.modal_hidden_size,
        )
        self.use_modal_start_token = self.config.use_modal_start_token
        self.use_modal_end_token = self.config.use_modal_end_token
        self.num_max_segment = self.config.text_encoder.params.get("num_segments", 2)

        self.mmbt = MMBTModel(self._mmbt_config, text_encoder, modal_encoder)

    def extract_modal_end_token(self, sample_list: Dict[str, Tensor]):
        # compute the position of the last non-masked token, which is <sep>
        gather_index = sample_list["input_mask"].sum(1, keepdim=True) - 1
        modal_end_token = (
            torch.gather(sample_list["input_ids"], 1, gather_index)
            .squeeze(1)
            .clone()
            .detach()
        )

        batch_size = sample_list["input_ids"].size(0)
        device = sample_list["input_ids"].device
        # remove start_token in input_ids
        sample_list["input_ids"] = torch.cat(
            [sample_list["input_ids"][:, 1:], sample_list["input_ids"][:, -1:]], dim=1
        )
        # update input_mask
        sample_list["input_mask"] = torch.cat(
            [
                sample_list["input_mask"][:, 1:],
                torch.zeros([batch_size, 1], dtype=torch.long, device=device),
            ],
            dim=1,
        )

        return modal_end_token

    def forward(self, sample_list: Dict[str, Tensor]):

        if self._is_direct_features_input:
            if "input_modal" in sample_list:
                input_modal = sample_list["input_modal"]
            else:
                input_modal = sample_list["image_feature_0"]
        else:
            input_modal = sample_list["image"]

        modal_start_token: Optional[Tensor] = None
        if self.use_modal_start_token:
            modal_start_token = sample_list["input_ids"][:, 0].clone().detach()

        modal_end_token: Optional[Tensor] = None
        if self.use_modal_end_token:
            modal_end_token = self.extract_modal_end_token(sample_list)

        if "modal_token_type_ids" in sample_list:
            modal_token_type_ids = sample_list["modal_token_type_ids"]
        else:
            token_value = 0
            segment_ids = sample_list["segment_ids"]
            max_id = segment_ids.max()
            min_id = segment_ids.min()
            # Case of only one segment
            if max_id == min_id:
                # If max_id is greater than 0, that means text is at 0 segment
                # which means modal will be at 1
                # In other case, it will be zero, which it already is
                # NOTE: We compare with tensor here due to TorchScript compliance
                if max_id == torch.tensor(0, dtype=max_id.dtype):
                    token_value = 1
            else:
                max_segment = self.num_max_segment - 1
                # If max id is not equal to max_segment, it means
                # text segments start from 0 which means modal will
                # be last, otherwise, it is 0, which it already is
                if max_id != torch.tensor(max_segment, dtype=max_id.dtype):
                    token_value = max_segment
            modal_token_type_ids = torch.full(
                (input_modal.size(0), 1),
                fill_value=token_value,
                dtype=torch.long,
                device=input_modal.device,
            )

        # In case of XRAY, there might be only two dims
        if input_modal.dim() == 2:
            input_modal = input_modal.unsqueeze(dim=1)

        # See details of inputs at
        # https://github.com/huggingface/transformers/blob/1789c7/src/transformers/modeling_mmbt.py#L101 # noqa
        output = self.mmbt(
            input_modal,
            input_ids=sample_list["input_ids"],
            modal_start_tokens=modal_start_token,
            modal_end_tokens=modal_end_token,
            attention_mask=sample_list["input_mask"],
            token_type_ids=sample_list["segment_ids"],
            modal_token_type_ids=modal_token_type_ids,
            position_ids=None,
            modal_position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        )

        return output


class MMBTForPreTraining(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.bert = MMBTBase(config, *args, **kwargs)
        self.encoder_config = self.bert.encoder_config

        # TODO : Switch to AutoModelForPreTraining after transformers
        # package upgrade to 2.5
        pretraining_module = BertForPreTraining.from_pretrained(
            self.config.bert_model_name,
            config=self.encoder_config,
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
        )

        self.cls = deepcopy(pretraining_module.cls)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we
            are cloning them instead.
        """
        if hasattr(self, "cls"):
            self.bert.mmbt.transformer._tie_or_clone_weights(
                self.cls.predictions.decoder,
                self.bert.mmbt.transformer.embeddings.word_embeddings,
            )

    def forward(self, sample_list):
        module_output = self.bert(sample_list)
        sequence_output, pooled_output = module_output[0], module_output[1]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        output = {}
        if (
            self.encoder_config.output_hidden_states
            or self.encoder_config.output_attentions
        ):
            output["extras"] = module_output[2:]

        loss_key = f"{sample_list.dataset_name}/{sample_list.dataset_type}"

        if "lm_label_ids" in sample_list and sample_list.lm_label_ids is not None:
            output["logits"] = prediction_scores
            lm_label_ids = sample_list.lm_label_ids
            # Only take last scores which are text's scores and ignore image scores
            text_scores = (
                prediction_scores[:, -(lm_label_ids.size(1)) :]
                .contiguous()
                .view(-1, self.encoder_config.vocab_size)
            )
            masked_lm_loss = self.loss_fct(
                text_scores, sample_list.lm_label_ids.contiguous().view(-1)
            )
            output["losses"] = {}
            output["losses"][f"{loss_key}/masked_lm_loss"] = masked_lm_loss

        # Add alignment loss if present
        if (
            "image_text_alignment" in sample_list
            and sample_list.image_text_alignment is not None
        ):
            output["seq_relationship_logits"] = seq_relationship_score
            alignment_loss = self.loss_fct(
                seq_relationship_score.contiguous().view(-1),
                sample_list.image_text_alignment.contiguous().view(-1),
            )
            output["losses"][f"{loss_key}/alignment_loss"] = alignment_loss

        return output


class MMBTForClassification(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.bert = MMBTBase(config, *args, **kwargs)
        self.encoder_config = self.bert.encoder_config
        self.num_labels = self.config.num_labels
        self.output_hidden_states = self.encoder_config.output_hidden_states
        self.output_attentions = self.encoder_config.output_attentions
        self.fused_feature_only = self.config.fused_feature_only

        self.dropout = nn.Dropout(self.encoder_config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.encoder_config),
            nn.Linear(self.encoder_config.hidden_size, self.config.num_labels),
        )

    def forward(self, sample_list: Dict[str, Tensor]):
        module_output = self.bert(sample_list)
        pooled_output = module_output[1]
        output = {}

        if not torch.jit.is_scripting():
            if self.output_hidden_states or self.output_attentions:
                output["extras"] = module_output[2:]
        else:
            assert not (
                self.output_hidden_states or self.output_attentions
            ), "output_attentions or output_hidden_states not supported in script mode"

        pooled_output = self.dropout(pooled_output)

        if self.fused_feature_only:
            output["fused_feature"] = self.classifier[0](pooled_output)
            return output

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output["scores"] = reshaped_logits

        return output


@registry.register_model("mmbt")
class MMBT(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        model: str = "mmbt"
        # classification or pretraining
        training_head_type: str = "pretraining"
        bert_model_name: str = "bert-base-uncased"
        direct_features_input: bool = False
        freeze_text: bool = False
        freeze_modal: bool = False
        freeze_complete_base: bool = False
        finetune_lr_multiplier: float = 1
        # Dimension of the embedding finally returned by the modal encoder
        modal_hidden_size: int = 2048
        text_hidden_size: int = 768
        num_labels: int = 2
        # This actually is Union[ImageEncoderConfig, ImageFeatureEncoderConfig]
        modal_encoder: EncoderFactory.Config = ImageEncoderFactory.Config(
            type=ImageEncoderTypes.resnet152, params=ResNet152ImageEncoder.Config()
        )
        text_encoder: EncoderFactory.Config = TextEncoderFactory.Config(
            type=TextEncoderTypes.transformer,
            params=TransformerEncoder.Config(bert_model_name=II("bert_model_name")),
        )
        use_modal_start_token: bool = True
        use_modal_end_token: bool = True
        fused_feature_only: bool = False
        output_dim: int = 768

    def __init__(self, config: Union[DictConfig, Config], *args, **kwargs):
        super().__init__(config)

    def build(self):
        if self.config.training_head_type == "pretraining":
            self.model = MMBTForPreTraining(self.config)
        else:
            self.model = MMBTForClassification(self.config)

        if self.config.freeze_complete_base or self.config.freeze_text:
            for p in self.model.bert.mmbt.transformer.parameters():
                p.requires_grad = False

        if self.config.freeze_complete_base or self.config.freeze_modal:
            for p in self.model.bert.mmbt.modal_encoder.parameters():
                p.requires_grad = False

    # Backward compatibility for code from older mmbt
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("base.bert", "model.bert")
            .replace("base.cls", "model.cls")
            .replace("base.classifier", "model.classifier")
        )

    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        model = super().from_pretrained(model_name, *args, **kwargs)
        config = load_pretrained_model(model_name)["full_config"]
        OmegaConf.set_struct(config, True)
        if model_name == "mmbt.hateful_memes.images" or kwargs.get("interface"):
            return MMBTGridHMInterface(model, config)
        return model

    @classmethod
    def config_path(cls):
        return "configs/models/mmbt/pretrain.yaml"

    def forward(self, sample_list: Dict[str, Tensor]):
        return self.model(sample_list)

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)
