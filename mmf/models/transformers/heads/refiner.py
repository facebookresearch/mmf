# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
from mmf.common.registry import registry
from mmf.models.transformers.base import BaseTransformerHead
from mmf.modules.losses import RefinerContrastiveLoss, RefinerMSLoss
from torch import nn

try:
    from transformers3.modeling_bert import BertOnlyMLMHead
except ImportError:
    from transformers.modeling_bert import BertOnlyMLMHead


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dims: List[int],
        dropout: float = 0.3,
        nonlinearity: Type[nn.Module] = nn.ReLU,
        normalization: Optional[Type[nn.Module]] = nn.LayerNorm,
    ):
        super().__init__()
        self.output_dim = mlp_dims[-1]
        projection_prev_dim = input_dim
        projection_modulelist = []

        for mlp_dim in mlp_dims:
            projection_modulelist.append(nn.Linear(projection_prev_dim, mlp_dim))
            projection_modulelist.append(nonlinearity())

            if normalization is not None:
                projection_modulelist.append(normalization(mlp_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))
            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        x = self.projection(x)
        return x


@registry.register_transformer_head("refiner")
class Refiner(BaseTransformerHead):
    @dataclass
    class Config(BaseTransformerHead.Config):
        type: str = "refiner"
        vocab_size: int = 30522
        hidden_size: int = 768
        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-5
        hidden_act: str = "gelu"
        ignore_index: int = -1
        loss_name: str = "refiner_ss_loss"
        loss_type: str = "mse"
        refiner_target_pooler: str = "average_k_from_last"
        refiner_target_layer_depth: int = 1
        label_key: Optional[str] = None
        modalities: List[str] = field(default_factory=lambda: ["text", "image"])
        weights: List[float] = field(default_factory=lambda: [0.1, 0.1])
        tol: float = 0.000001

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.cls = BertOnlyMLMHead(self.config)
        loss_dict = dict(
            mse=torch.nn.MSELoss(),
            cosine=torch.nn.CosineSimilarity(dim=1),
            contrastive=RefinerContrastiveLoss(),
            ms=RefinerMSLoss(),
        )
        self.refiner_loss = loss_dict.get(self.config.loss_type)

        self.refiner_decoder = {}
        self.weights = {}
        for i, modality in enumerate(self.config.modalities):
            self.refiner_decoder[modality] = MLP(
                input_dim=self.config.hidden_size,
                mlp_dims=[self.config.hidden_size],
                dropout=self.config.hidden_dropout_prob,
                nonlinearity=torch.nn.ReLU,
                normalization=torch.nn.LayerNorm,
            )
            self.weights[modality] = self.config.weights[i]

        self.modalities = self.config.modalities
        self.tol = self.config.tol
        self.refiner_target_pooler = self.config.refiner_target_pooler
        self.refiner_target_layer_depth = self.config.refiner_target_layer_depth
        self.loss_name = self.config.loss_name

        pool_class = registry.get_pool_class(self.refiner_target_pooler)
        if pool_class is None:
            raise ValueError(
                f"No pooler {self.refiner_target_pooler} is\
                             registered to registry"
            )
        self.pooler = pool_class(self.refiner_target_layer_depth)

    def forward(
        self,
        sequence_output: torch.Tensor,
        encoded_layers: Optional[List[torch.Tensor]] = None,
        processed_sample_list: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        start_token = {}
        end_token = {}
        prev_end_token = 0
        masks = []
        for modality in self.modalities:
            masks.append(processed_sample_list["masks"][modality])
            sz = processed_sample_list["masks"][modality].size()
            start_token[modality] = prev_end_token
            end_token[modality] = start_token[modality] + sz[1] - 1
            prev_end_token = end_token[modality] + 1

        pad_mask = torch.cat(masks, dim=1)
        processed_sample_list["refiner_outputs"] = {}
        processed_sample_list["refiner_outputs"]["fused_embedding"] = self.pooler(
            encoded_layers, pad_mask
        )
        processed_sample_list["refiner_targets"] = {}
        for modality in self.modalities:
            modality_encodings = []
            tk_start = start_token[modality]
            tk_end = end_token[modality]
            for enc_layers in encoded_layers:
                modality_encodings.append(enc_layers[:, tk_start : tk_end + 1, :])
            modality_mask_encodings = pad_mask[:, tk_start : tk_end + 1]
            processed_sample_list["refiner_targets"][modality] = self.pooler(
                modality_encodings, modality_mask_encodings
            )

        output_dict = {}
        prediction = self.cls(sequence_output)
        output_dict["logits"] = prediction
        output_dict["losses"] = {}

        fused_embedding = processed_sample_list["refiner_outputs"]["fused_embedding"]
        refiner_reconstruct = {}
        for modality in processed_sample_list["refiner_targets"].keys():
            local_device = fused_embedding.device
            self.refiner_decoder[modality].to(local_device)
            refiner_reconstruct[modality] = self.refiner_decoder[modality](
                fused_embedding
            )
            if isinstance(self.refiner_loss, torch.nn.CosineSimilarity):
                loss = self.weights[modality] * (
                    1.0
                    - torch.mean(
                        self.refiner_loss(
                            processed_sample_list["refiner_targets"][modality],
                            refiner_reconstruct[modality],
                        )
                    )
                )
            elif isinstance(self.refiner_loss, RefinerContrastiveLoss) or isinstance(
                self.refiner_loss, RefinerMSLoss
            ):
                modality_targets = {}
                modality_targets["targets"] = processed_sample_list["refiner_targets"][
                    modality
                ]
                refiner_modal_outputs = {}
                refiner_modal_outputs["scores"] = refiner_reconstruct[modality]
                loss = self.refiner_loss(modality_targets, refiner_modal_outputs)

            else:
                loss = self.weights[modality] * self.refiner_loss(
                    processed_sample_list["refiner_targets"][modality],
                    refiner_reconstruct[modality],
                )

            if "current_loss" not in locals():
                current_loss = loss
            else:
                current_loss += loss

        output_dict["losses"][self.loss_name] = current_loss
        output_dict["fused_embedding"] = fused_embedding
        return output_dict
