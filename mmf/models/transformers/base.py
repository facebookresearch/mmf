# Copyright (c) Facebook, Inc. and its affiliates.

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Type

from mmf.models import BaseModel
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from omegaconf import OmegaConf
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel


@dataclass
class BaseTransformerInput:
    input_ids: Dict[str, Tensor]  # dict of input ids for all modalities
    position_ids: Dict[str, Tensor]  # dict of position ids for all modalities
    segment_ids: Dict[str, Tensor]  # dict of segment/token type ids for all modalities
    masks: Dict[str, Tensor]  # dict of masks for all modalities


@dataclass
class BaseModalityConfigType:
    type: str  # type of modality, text, image, video, audio etc
    key: str  # name of modality
    # segment id to be used for modality. Each modality sould have different segment ids
    segment_id: int
    embedding_dim: int  # input dimension for modality embedding
    position_dim: int  # input dimension for position embedding
    # eps for layer norm, default is base transformer layer_norm_eps
    layer_norm_eps: float
    # dropout probability, default is base transformer hidden_dropout_prob
    hidden_dropout_prob: float


@dataclass
class BaseTransformerConfigType:
    transformer_base: str  # name of transformer base model
    training_head_type: str  # training head type used for initializing head
    modalities: List[BaseModalityConfigType]  # list of modalities for the model input
    initializer_range: float  # std dev of the normal distribution to initialize layers
    initializer_mean: float  # mean of the normal distribution to initialize layers
    token_noise_std: float  # mean of normal noise for token embeddings
    token_noise_mean: float  # stddev of normal noise for token embeddings
    layer_norm_weight_fill: float  # layer norm weight initialization
    random_initialize: bool  # random initialize whole network
    freeze_transformer: bool  # freeze the base transformer
    finetune_lr_multiplier: float  # finetune lr multiplier for base transformer


class BaseTransformer(BaseModel):
    def __init__(self, config: BaseTransformerConfigType):
        """Initialize the config which is the model configuration and transformer_config
        which is the config for the `transformer` base model.
        """
        super().__init__(config)
        self.config = config
        self.transformer_config = AutoConfig.from_pretrained(
            config.transformer_base, **OmegaConf.to_container(config)
        )

    def build(self):
        """Build the different parts of the multimodal transformer model and
        initializes weights.
        """
        self.build_transformer()
        self.build_encoders()
        self.build_embeddings()
        self.build_heads()
        self.build_losses()

        self.init_weights()

    def get_optimizer_parameters(self, config: BaseTransformerConfigType):
        return get_optimizer_parameters_for_bert(self, config)

    def build_encoders(self):
        """Build any encoders for different input modalities. Encoders are used while
        preprocessing a sample. We the visual_encoder by default for raw image input.

        Example ::

            # For image
            self.image_encoder = ImageEncoder(self.config)

        """
        return

    def build_embeddings(self):
        """Build the embeddings for the different input modalities.

        Example ::

            # For text
            self.word_embeddings = nn.Embedding(
                config.vocab_size, config.hidden_size, padding_idx=0
            )
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )

            # For image
            self.img_embeddings = nn.Sequential(
                nn.Linear(img_dim, config.hidden_size),
                torch.nn.LayerNorm(config.hidden_size, eps=1e-12),
            )
        """
        return

    def build_transformer(self):
        """Build the transformer encoder. This uses transformers AutoModel to load a
        pretrained model given the name of the transformer based model. All the layers
        in the transformer model will be available (encoder, embeddings etc.) for use.
        Different HUggingface transformer models have different naming conventions for
        the layers. Adjust your derived class based on the transformer base you want to
        use.

        Example ::

            self.transformer = AutoModel.from_pretrained(
                "bert-base-uncased",
                config=self.transformer_config,
            )
        """
        self.transformer = AutoModel.from_pretrained(
            self.config.transformer_base,
            config=self.transformer_config,
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
        )

    def build_heads(self):
        """Build the different heads for the model. It can be either the pretraining
        head or the classifier heads.

        Example ::

            # For pretraining
            self.classifier = nn.Sequential(
                BertPredictionHeadTransform(self.transformer_config),
                nn.Linear(self.transformer_config.hidden_size, self.config.num_labels),
            )

            # For classification
            self.classifier = nn.Sequential(
                BertPredictionHeadTransform(self.transformer_config),
                nn.Linear(self.transformer_config.hidden_size, self.config.num_labels),
            )
        """
        return

    def build_losses(self):
        """Initialize the losses for pretraining. For example MLM, MIM etc.

        Example ::

            self.mlm_loss = CrossEntropyLoss(ignore_index=-1)
        """
        return

    def _init_weights(self, module: Type[nn.Module]):
        """Initialize the weights for different layers.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=self.config.initializer_mean, std=self.config.initializer_range
            )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.layer_norm_weight_fill)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings
        if required.
        """
        return

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.config.transformer_base is None:
                # No pretrained model, init weights
                self.apply(self._init_weights)

        # Tie weights if required
        self.tie_weights()

    def preprocess_sample(self, sample_list: Dict[str, Any]) -> BaseTransformerInput:
        """Preprocess the sample_list and returns input ids, position ids, segment or
        token type ids and masks for different modalities.

        Returns:
            BaseTransformerInput: BaseTransformerInput containing input_ids,
                position_ids, segment_ids, masks
        """
        return

    def forward(self, sample_list: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Forward pass of the model. The input sample_list can be preprocessed using
        the preprocess_sample method which expects to return a BaseTransformerInput
        object. BaseTransformerInput contains different properties of the input
        modalities and the masks. These can be used to generate embeddings for each
        modality and also create attention mask.

        Flow of how the forward pass can be implemented using various modules in
        BaseTransformer:

                    preprocess_sample                          ||
                            |                                  ||
                   generate embeddings                         ||
                            |                                  ||
                 generate attention masks                      ||     MODEL
                            |                                  ||
                 transformer encoder pass                      ||     FLOW
                            |                                  ||
                   different head pass                         ||   DIRECTION
                            |                                  ||
                   postprocess_output                          ||
                            |                                  ||
                 Dict[str, Tensor] output                      \/

        Returns:
            Dict[str, Tensor]: Dict containing scores or losses
        """
        return

    def postprocess_output(self, output: List[Tensor]) -> Dict[str, Tensor]:
        """Postprocessing the output from the transformer head, for pretraining
        it's the output of the pretrain head and for classification its the output
        of the classsification head. Calculate lossses on pretraining output or
        model output scores.

        Returns:
            Dict[str, Tensor]: Dict containing scores or losses
        """
        return output
