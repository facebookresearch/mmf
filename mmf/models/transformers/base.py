# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type

from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.encoders import IdentityEncoder
from mmf.utils.modeling import get_bert_configured_parameters
from omegaconf import MISSING, OmegaConf
from torch import nn, Tensor


logger = logging.getLogger(__name__)


class BaseTransformerInput(NamedTuple):
    input_ids: Dict[str, Tensor]  # dict of input ids for all modalities
    position_ids: Dict[str, Tensor]  # dict of position ids for all modalities
    segment_ids: Dict[str, Tensor]  # dict of segment/token type ids for all modalities
    masks: Dict[str, Tensor]  # dict of masks for all modalities


@dataclass
class BaseTransformerModalityConfig:
    type: str = MISSING  # type of modality, text, image, video, audio etc
    key: str = MISSING  # name of modality
    # segment id to be used for modality. Each modality sould have different segment ids
    segment_id: int = MISSING
    embedding_dim: int = MISSING  # input dimension for modality embedding
    position_dim: int = MISSING  # input dimension for position embedding
    # eps for layer norm, default is base transformer layer_norm_eps
    layer_norm_eps: float = 1e-12
    # dropout probability, default is base transformer hidden_dropout_prob
    hidden_dropout_prob: float = 0.1
    # Encoder to be used to encode this particular modality
    # This is actually: Union[EncoderFactory.Config, Encoder.Config]
    # NOTE: Waiting on https://github.com/omry/omegaconf/issues/144
    encoder: Any = IdentityEncoder.Config()
    # when type is text, whether to consume raw text or intermediate representations
    # from frozen text encoder. This can be potentially also used by other modalities.
    consume_raw: bool = True


@dataclass
class BaseTransformerBackendConfig:
    # Type of the backend, e.g. huggingface
    type: str = MISSING
    # Whether to freeze the backend parameters
    freeze: bool = False
    # Parameters for the backend
    params: Dict[str, Any] = field(default_factory=lambda: {})


class BaseTransformer(BaseModel):
    # NOTE: Please define the values for the config parameters
    # in your inherited class
    @dataclass
    class Config(BaseModel.Config):
        # registry key of the model
        model: str = MISSING
        # name of transformer base model
        transformer_base: str = MISSING
        # training head type used for initializing head
        training_head_type: str = MISSING
        # backend of the transformer
        backend: BaseTransformerBackendConfig = MISSING
        # list of modalities for the model input
        modalities: List[BaseTransformerModalityConfig] = MISSING
        # std dev of the normal distribution to initialize layers
        initializer_range: float = MISSING
        # mean of the normal distribution to initialize layers
        initializer_mean: float = MISSING
        # mean of normal noise for token embeddings
        token_noise_std: float = MISSING
        # stddev of normal noise for token embeddings
        token_noise_mean: float = MISSING
        # layer norm weight initialization
        layer_norm_weight_fill: float = MISSING
        # random initialize whole network
        random_initialize: bool = MISSING
        # freeze the base transformer
        freeze_transformer: bool = MISSING
        # finetune lr multiplier for base transformer
        finetune_lr_multiplier: float = MISSING

    def __init__(self, config: Config):
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

    def get_optimizer_parameters(self, config):
        lr = config.optimizer.params.lr

        param_list = []
        parameters = []
        head_configs = self.config.get("heads", [])
        for name, module in self.named_children():
            # Heads can have different learning rates. This is handled here
            if name == "heads":
                # Parameters in the head which have a separate learning
                # rate, are added as a separate param group
                for head_config, head in zip(head_configs, self.heads):
                    parameters, param_list = self.set_lr_for_parameters(
                        config=head_config,
                        module_name="{} head".format(head_config.get("type", "MLP")),
                        base_lr=lr,
                        module=head,
                        parameters=parameters,
                        param_list=param_list,
                    )
            elif name == "encoders":
                for key in module:
                    for modality in self.config.modalities:
                        if key == modality.key:
                            modality_config = modality
                    parameters, param_list = self.set_lr_for_parameters(
                        config=modality_config,
                        module_name=f"{key} encoder",
                        base_lr=lr,
                        module=module[key],
                        parameters=parameters,
                        param_list=param_list,
                    )
            else:
                # For other modules in trunk, add to same param group
                param_list += list(module.named_parameters())

        parameters += get_bert_configured_parameters(param_list)

        return parameters

    def set_lr_for_parameters(
        self, config, module_name, base_lr, module, parameters, param_list
    ):
        lr_multiplier = config.get("lr_multiplier", 1.0)
        if lr_multiplier != 1.0:
            logger.info(
                f"Setting learning rate of {module_name} to be "
                f"{base_lr} * {lr_multiplier}."
            )  # noqa
            parameters += get_bert_configured_parameters(
                module, base_lr * lr_multiplier
            )
        else:
            # Parameters for the modules with same learning rate as
            # trunk, add to same param group
            param_list += list(module.named_parameters())
        return parameters, param_list

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
        backend_type = backend_config.get("type", "huggingface")
        backend_class = registry.get_transformer_backend_class(backend_type)
        self.backend = backend_class(self.config)

        if backend_config.get("freeze", False):
            for param in self.backend.parameters():
                param.requires_grad = False

    def build_heads(self):
        """Build the different heads for the model. It can be either the pretraining
        head or the classifier heads.
        """
        self.heads = nn.ModuleList()
        head_configs = self.config.get("heads", [])
        for head_config in head_configs:
            head_type = head_config.get("type", "mlp")
            head_class = registry.get_transformer_head_class(head_type)
            self.heads.append(head_class(head_config))

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

    def preprocess_sample(
        self, sample_list: Dict[str, Any]
    ) -> Dict[str, Dict[str, Tensor]]:
        """Preprocess the sample_list and returns input ids, position ids, segment or
        token type ids and masks for different modalities.

        Returns:
            Dict[str, Dict[str, Tensor]]: containing input_ids, position_ids,
                segment_ids, masks
        """
        return

    def forward(self, sample_list: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Forward pass of the model. The input sample_list can be preprocessed using
        the preprocess_sample method which expects to return a
        Dict[str, Dict[str, Tensor]] object. It contains different properties of the
        input modalities and the masks. These can be used to generate embeddings for
        each modality and also create attention mask.

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


class BaseTransformerBackend(nn.Module, ABC):
    def __init__(self, config: BaseTransformer.Config, *args, **kwargs):
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


class BaseTransformerHead(nn.Module, ABC):
    @dataclass
    class Config:
        type: str = MISSING
        # Whether to freeze the head parameters
        freeze: bool = False
        # LR multiplier for the head, (head_lr = base_lr * lr_multiplier)
        lr_multiplier: float = 1.0

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = OmegaConf.create({**asdict(self.Config()), **config})

    @classmethod
    def from_params(cls, **kwargs):
        config = OmegaConf.structured(cls.Config(**kwargs))
        return cls(config)

    def tie_weights(self, module: Optional[nn.Module] = None):
        pass

    @abstractmethod
    def forward(
        self,
        sequence_output: Tensor,
        encoded_layers: Optional[List[Tensor]] = None,
        processed_sample_list: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ) -> Dict[str, Tensor]:
        """Forward for the head module.

        Warning: Empty shell for code to be implemented in other class.
        """
