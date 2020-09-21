# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
from typing import Any, Dict, List, Type

import torch
from mmf.common.registry import registry
from mmf.common.typings import DictConfig
from mmf.models.transformers.base import (
    BaseTransformer,
    BaseTransformerConfigType,
    BaseTransformerInput,
)
from mmf.modules.encoders import MultiModalEncoderBase
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertForPreTraining,
    BertPooler,
    BertPredictionHeadTransform,
)


class ImageEncoder(MultiModalEncoderBase):
    """Extends the MultiModalEncoderBase class which builds the encoder based on
    the config parameters. We can set the type of image encoder(resnet50, resnet152,
    resnext50 etc) and other parameters like num of features, type of pooling etc.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def build(self):
        self.encoder = self._build_modal_encoder(self.config.image_encoder)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class MMFTransformerEmbeddings(nn.Module):
    """Embedding class that can take any number of image or text modalities, each can
    have their input id, position id and segment id. We generate embeddings of
    dimension config.hidden_size for each and then first add the three embeddings
    for each modality to have a modality specific embedding. We then concat the
    modality specific embeddings to have a joint embedding.
    """

    def __init__(
        self,
        model_config: BaseTransformerConfigType,
        transformer_config: Dict[str, Any],
        transformer: Type[nn.Module],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.model_config = model_config
        self.transformer_config = transformer_config

        self.token_embeddings = nn.ModuleList()
        self.pos_embeddings = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.modality_keys: List = []

        # Build layers for each modality and initialize
        self.build_layers()
        self.init_weights(transformer)

        assert (
            len(self.token_embeddings)
            == len(self.pos_embeddings)
            == len(self.layer_norms)
            == len(self.dropouts)
            == len(self.model_config.modalities)
        )

    def build_layers(self):

        for modality in self.model_config.modalities:
            self.modality_keys.append(modality.key)
            layer_norm_eps = modality.get(
                "layer_norm_eps", self.transformer_config.layer_norm_eps
            )
            position_dim = modality.get(
                "position_dim", self.transformer_config.max_position_embeddings
            )
            hidden_dropout_prob = modality.get(
                "hidden_dropout_prob", self.transformer_config.hidden_dropout_prob
            )
            if modality.type == "text":
                self.token_embeddings.append(
                    nn.Embedding(
                        self.transformer_config.vocab_size,
                        self.transformer_config.hidden_size,
                        padding_idx=self.transformer_config.pad_token_id,
                    )
                )
            elif modality.type == "image":
                self.token_embeddings.append(
                    nn.Sequential(
                        nn.Linear(
                            modality.embedding_dim, self.transformer_config.hidden_size
                        ),
                        torch.nn.LayerNorm(
                            self.transformer_config.hidden_size, eps=layer_norm_eps
                        ),
                    )
                )
            self.pos_embeddings.append(
                nn.Embedding(position_dim, self.transformer_config.hidden_size)
            )
            self.layer_norms.append(
                torch.nn.LayerNorm(
                    self.transformer_config.hidden_size, eps=layer_norm_eps
                )
            )
            self.dropouts.append(nn.Dropout(hidden_dropout_prob))

        self.token_type_embeddings = nn.Embedding(
            len(self.model_config.modalities), self.transformer_config.hidden_size
        )

    def init_weights(self, transformer: Type[nn.Module]):
        for idx, modality in enumerate(self.model_config.modalities):
            if modality.type == "text":
                self.token_embeddings[idx] = transformer.embeddings.word_embeddings
                self.layer_norms[idx] = transformer.embeddings.LayerNorm

            self.pos_embeddings[idx].weight = nn.Parameter(
                deepcopy(transformer.embeddings.position_embeddings.weight.data),
                requires_grad=True,
            )

        # Token Type or Segment Embeddings
        if hasattr(transformer.embeddings, "token_type_embeddings"):
            token_vocab_size = self.transformer_config.type_vocab_size
            self.token_type_embeddings.weight.data[:token_vocab_size].copy_(
                transformer.embeddings.token_type_embeddings.weight
            )
            for idx in range(token_vocab_size, len(self.model_config.modalities)):
                self.token_type_embeddings.weight.data[idx].copy_(
                    transformer.embeddings.token_type_embeddings.weight.data.mean(dim=0)
                )
                # Add random normal noise
                self.token_type_embeddings.weight.data[idx] += torch.normal(
                    self.model_config.token_noise_mean,
                    self.model_config.token_noise_std,
                    size=self.token_type_embeddings.weight.data[idx].size(),
                )

    def forward(
        self,
        input_ids: Dict[str, Tensor],
        position_ids: Dict[str, Tensor],
        segment_ids: Dict[str, Tensor],
    ) -> Tensor:
        list_embeddings = []
        for idx, (token_emb, pos_emb, layer_norm, dropout) in enumerate(
            zip(
                self.token_embeddings,
                self.pos_embeddings,
                self.layer_norms,
                self.dropouts,
            )
        ):
            modality_name = self.modality_keys[idx]
            total_embedding = token_emb(input_ids[modality_name])
            if modality_name in position_ids:
                total_embedding += pos_emb(position_ids[modality_name])

            if modality_name in segment_ids:
                total_embedding += self.token_type_embeddings(
                    segment_ids[modality_name]
                )

            list_embeddings.append(dropout(layer_norm(total_embedding)))

        return torch.cat(list_embeddings, dim=1)


@registry.register_model("mmf_transformer")
class MMFTransformer(BaseTransformer):
    def __init__(self, config: BaseTransformerConfigType, *args, **kwargs):
        super().__init__(config)
        self.num_labels = self.config.num_labels
        self.vocab_size = self.transformer_config.vocab_size
        self.modality_keys: List = []
        for modality in self.config.modalities:
            self.modality_keys.append(modality.key)

    @classmethod
    def config_path(cls) -> str:
        return "configs/models/mmf_transformer/defaults.yaml"

    def build_encoders(self):
        self.image_encoder = ImageEncoder(self.config)
        if getattr(self.config, "freeze_image_encoder", False):
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def build_embeddings(self):
        """Initialize the embedding class we will use for multiple
        modalities (here just text and image). For the text embeeddings we will use the
        pretrained weights from the trasnformer model rather than training from scratch.
        """
        self.embeddings = MMFTransformerEmbeddings(
            self.config, self.transformer_config, self.transformer
        )

    def build_heads(self):
        """Initialize the classifier head. It takes the output of the
        transformer encoder and passes it through a pooler (we use the pooler from BERT
        model), then dropout, BertPredictionHeadTransform (which is a linear layer,
        followed by activation and layer norm) and lastly a linear layer projecting the
        hidden output to classification labels.
        """
        if self.config.training_head_type == "classification":
            self.classifier = nn.Sequential(
                BertPooler(self.transformer_config),
                nn.Dropout(self.transformer_config.hidden_dropout_prob),
                BertPredictionHeadTransform(self.transformer_config),
                nn.Linear(self.transformer_config.hidden_size, self.config.num_labels),
            )
        elif self.config.training_head_type == "pretraining":
            module = BertForPreTraining.from_pretrained(
                self.config.transformer_base, config=self.transformer_config
            )
            self.pooler = BertPooler(self.transformer_config)
            self.classifier = module.cls

    def build_losses(self):
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def preprocess_sample(self, sample_list: Dict[str, Any]) -> BaseTransformerInput:
        """Preprocess the sample list elements and form a BaseTransformerInput
        type object. This object standardizes how we represent multiple modalities.
        Check the definition of this dataclass in BaseTransformer.
        """

        # Input IDs (or text tokens/image features)
        input_ids: Dict[str, Tensor] = {}
        for idx, modality in enumerate(self.config.modalities):
            if modality.type == "text":
                if sample_list.input_ids.dim() > 2:
                    input_ids[modality.key] = sample_list.input_ids[:, idx]
                else:
                    input_ids[modality.key] = sample_list.input_ids
            elif modality.type == "image":
                if "image" in sample_list:
                    image_modal = sample_list.image
                else:
                    image_modal = sample_list.image_feature_0
                input_ids[modality.key] = self.image_encoder(image_modal)

        # Position IDs
        position_ids: Dict[str, Tensor] = {}
        for modality in self.config.modalities:
            position_ids[modality.key] = (
                torch.arange(
                    0,
                    input_ids[modality.key].size(1),
                    dtype=torch.long,
                    device=input_ids[modality.key].device,
                )
                .unsqueeze(0)
                .expand(input_ids[modality.key].size()[:2])
            )

        # Segment IDs
        segment_ids: Dict[str, Tensor] = {}
        for idx, modality in enumerate(self.config.modalities):
            if modality.type == "text" and hasattr(sample_list, "segment_ids"):
                if sample_list.segment_ids.dim() > 2:
                    segment_ids[modality.key] = sample_list.segment_ids[:, idx]
                else:
                    segment_ids[modality.key] = sample_list.segment_ids
            elif hasattr(modality, "segment_id"):
                segment_ids[modality.key] = torch.zeros(
                    input_ids[modality.key].size()[:2],
                    dtype=torch.long,
                    device=input_ids[modality.key].device,
                ).fill_(modality.segment_id)

        # Masks
        masks: Dict[str, Tensor] = {}
        for idx, modality in enumerate(self.config.modalities):
            if modality.type == "text":
                if sample_list.input_mask.dim() > 2:
                    masks[modality.key] = sample_list.input_mask[:, idx]
                else:
                    masks[modality.key] = sample_list.input_mask

            elif modality.type == "image":
                if "image_mask" in sample_list:
                    masks[modality.key] = sample_list.image_mask
                else:
                    masks[modality.key] = torch.ones(
                        input_ids[modality.key].size()[:-1],
                        dtype=torch.long,
                        device=input_ids[modality.key].device,
                    )

        masked_lm_labels: Dict[str, Tensor] = {}
        for idx, modality in enumerate(self.config.modalities):
            if modality.type == "text":
                if sample_list.lm_label_ids.dim() > 2:
                    masked_lm_labels[modality.key] = sample_list.lm_label_ids[:, idx]
                else:
                    masked_lm_labels[modality.key] = sample_list.lm_label_ids
            else:
                masked_lm_labels[modality.key] = torch.zeros(
                    input_ids[modality.key].size()[:2],
                    dtype=torch.long,
                    device=input_ids[modality.key].device,
                ).fill_(-1)

        return BaseTransformerInput(
            input_ids, position_ids, segment_ids, masks, masked_lm_labels
        )

    def transformer_encode(self, embedding, attention_mask):
        return self.transformer.encoder(
            embedding, attention_mask, [None] * len(self.transformer.encoder.layer)
        )

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Sample preprocess
        output = self.preprocess_sample(sample_list)

        # Transformer Input Embeddings
        embedding_output = self.embeddings(
            input_ids=output["input_ids"],
            position_ids=output["position_ids"],
            segment_ids=output["segment_ids"],
        )

        # Transformer Attention mask
        # concat the attention masks for all modalities
        masks = []
        for modality in self.modality_keys:
            masks.append(output["masks"][modality])
        attention_mask = torch.cat(masks, dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Transformer Encoder
        encoded_layers = self.transformer_encode(
            embedding_output,  # combined embedding
            extended_attention_mask,  # combined attention mask
        )

        # Transformer Heads

        prediction_scores, seq_relationship_score = self.classifier(
            encoded_layers[0], self.pooler(encoded_layers[0])
        )
        # head_output = self.classifier(encoded_layers[0])

        # Postprocess outputs
        return self.postprocess_output(prediction_scores, output)

    def postprocess_output(
        self, output: Tensor, input: BaseTransformerInput
    ) -> Dict[str, Tensor]:
        """Postprocess the output from the classifier head and reshape it.
        This will be used to calculate losses and metrics in mmf.
        """
        output_dict = {}
        if self.config.training_head_type == "classification":
            output_dict["scores"] = output.contiguous().view(-1, self.num_labels)
        elif self.config.training_head_type == "pretraining":
            output_dict["logits"] = output
            masked_labels = []
            for modality in self.modality_keys:
                masked_labels.append(input.masked_lm_labels[modality])

            masked_labels = torch.cat(masked_labels, dim=-1)
            masked_lm_loss = self.ce_loss(
                output.contiguous().view(-1, self.vocab_size),
                masked_labels.contiguous().view(-1),
            )
            output_dict["losses"] = {}
            output_dict["losses"]["masked_lm_loss"] = masked_lm_loss
        return output_dict
