# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Tuple, Type

from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from torch import Tensor, nn


class BaseTransformerInput(NamedTuple):
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
    finetune_lr_multiplier: float  # finetune lr multiplier for base transformer


class BaseTransformerBackend(nn.Module, ABC):
    def __init__(self, config: BaseTransformerConfigType, *args, **kwargs):
        super().__init__()
        self.config = config
        self.build_transformer_config()
        self.build_transformer_base()
        self.build_embeddings()

    @abstractmethod
    def build_transformer_config(self):
        """Build the transformer base model config.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def build_transformer_base(self):
        """Build the transformer base model.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def build_embeddings(self):
        """Build the multimodal embeddings using the transformer base
        embeddings.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def get_config(self):
        """Return the transformer configuration. This can be the config built
        in `build_transformer_config` or the model config passed to init.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def generate_embeddings(
        self,
        tokens_ids: Dict[str, Tensor],
        position_ids: Dict[str, Tensor],
        segment_ids: Dict[str, Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        """Generate the multimodal embeddings.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def generate_attention_mask(self, masks: List[Tensor]) -> Tensor:
        """Generate attention mask.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def generate_encoded_layers(self, embedding, attention_mask) -> List[Tensor]:
        """Generate the output from transformer layers. Return the encoded layers.

        Warning: Empty shell for code to be implemented in other class.
        """

    def forward(
        self,
        tokens_ids: Dict[str, Tensor],
        position_ids: Dict[str, Tensor],
        segment_ids: Dict[str, Tensor],
        masks: List[Tensor],
    ) -> Tuple[Tensor, List[Tensor]]:
        # Attention mask
        attention_mask = self.generate_attention_mask(masks)

        # Multimodal Embeddings
        embedding = self.generate_embeddings(
            tokens_ids, position_ids, segment_ids, attention_mask
        )

        # Encoder
        encoded_layers = self.generate_encoded_layers(embedding, attention_mask)

        # Output Tuple(sequence output, all encoded layers)
        return encoded_layers[-1], encoded_layers


class BaseTransformer(BaseModel):
    def __init__(self, config: BaseTransformerConfigType):
        """Initialize the config which is the model configuration and transformer_config
        which is the config for the `transformer` base model.
        """
        super().__init__(config)
        self.config = config

    def build(self):
        """Build the different parts of the multimodal transformer model and
        initializes weights.
        """
        self.build_backend()
        self.build_encoders()
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

    def build_backend(self):
        """Build the transformer backend. Use the `BaseTransformerBackend` base class
        to inherit from when building a new backend. All the layers in the transformer
        backend model will be available (encoder, embeddings etc.) for use. Adjust
        your derived class based on the transformer backend you want to use.
        """
        backend_config = self.config.get("backend", {})
        backend_type = getattr(backend_config, "type", "huggingface")
        backend_class = registry.get_transformer_backend_class(backend_type)
        self.backend = backend_class(self.config)

        if backend_config.get("freeze", False):
            for param in self.backend.parameters():
                param.requires_grad = False

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
        """Initialize the weights for different layers."""
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
