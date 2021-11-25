# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import collections.abc
from typing import Dict, List, Optional, Union

from mmf.common.registry import registry
from torch import Tensor, nn


def build_heads_dict(head_configs: Union[Dict, List], tasks: List, losses: Dict):
    """
    HeadsDict static constructor.
    This function either,
        returns a list of heads if head_configs is a list,
        returns a dict of task: [ head1, head2, ... ] if head_configs is a dict

        loss_names are a list or dict describing the loss module used for each head
        loss_names has the same shape as heads

        head_names is a list or dict containing head name strings
        head_names is used to describe bad heads in exceptions
    """

    def head_from_config(config):
        head_type = config.get("type", "mlp")
        head_class = registry.get_transformer_head_class(head_type)
        return head_class(config)

    if isinstance(head_configs, collections.abc.Sequence):
        heads = nn.ModuleList(
            [head_from_config(head_conf) for head_conf in head_configs]
        )
        head_loss_names = [head_conf.get("loss") for head_conf in head_configs]
        head_names = [head_conf.get("type", "mlp") for head_conf in head_configs]

    if isinstance(head_configs, collections.abc.Mapping):
        heads = nn.ModuleDict()
        head_names = {}  # used to describe head in exceptions
        head_loss_names = {}

        for task in tasks:
            head_config = head_configs.get(task)
            if head_config is None:
                raise ValueError(
                    f"No head defined for {task}. Dataset task {task} "
                    + "requires a head to return dict with 'losses'"
                )

            head_config_list = (
                head_config
                if isinstance(head_config, collections.abc.Sequence)
                else [head_config]
            )

            heads[task] = nn.ModuleList(
                [head_from_config(head_conf) for head_conf in head_config_list]
            )
            head_loss_names[task] = [
                head_conf.get("loss") for head_conf in head_config_list
            ]
            head_names[task] = [
                head_conf.get("type", "mlp") for head_conf in head_config_list
            ]

    return HeadsDict(heads, head_names, losses, head_loss_names)


class HeadsDict(nn.Module):
    """
    HeadsDict class manages the construction and forward pass for
    multiple possible heads for multi-task learning.
    Construction from list or dict configs is supported,
    take a look at `build_heads_dict(head_configs, tasks, losses)`.
    """

    def __init__(
        self,
        heads: Union[nn.ModuleDict, nn.ModuleList],
        head_names: Union[Dict, List],
        losses: Dict,
        head_loss_names: Union[Dict, List],
    ):
        super().__init__()
        self.heads = heads
        self.head_names = head_names
        self.losses = losses
        self.head_loss_names = head_loss_names

    def forward(
        self, task: Optional[str], sequence: Tensor, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        For a given task, compute the forward for each head
        associated with the task, compute the losses for
        each head, and sum the losses and scores
        """
        if isinstance(self.heads, nn.ModuleList):
            heads_modules_list = self.heads
            # list of losses, head_losses[i] is the loss name for outputs_list[i]
            head_losses = self.head_loss_names
            head_names = self.head_names
        else:
            heads_modules_list = self.heads[task]
            head_losses = self.head_loss_names[task]
            head_names = self.head_names[task]

        # list of dict( head outputs )
        outputs_list = [
            head(sequence, processed_sample_list=sample_list)
            for head in heads_modules_list
        ]

        assert len(head_losses) == len(outputs_list)

        # list of dict( losses, scores )
        processed_outputs_list = [
            self._process_head_output(outputs, loss_name, head_name, sample_list)
            for outputs, loss_name, head_name in zip(
                outputs_list, head_losses, head_names
            )
        ]

        def reduce_losses(accum_result, loss_dict):
            for loss_key, loss_val in loss_dict.items():
                if loss_key in accum_result:
                    accum_result[loss_key] += loss_val
                else:
                    accum_result[loss_key] = loss_val

        loss_result = {}
        for output in processed_outputs_list:
            reduce_losses(loss_result, output["losses"])

        results = {
            "losses": loss_result,
            "scores": sum(
                [output.get("scores", 0) for output in processed_outputs_list]
            ),
        }
        return results

    def _process_head_output(
        self,
        outputs: Union[Dict, Tensor],
        loss_name: str,
        head_name: str,
        sample_list: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        if isinstance(outputs, collections.MutableMapping) and "losses" in outputs:
            return outputs

        if isinstance(outputs, collections.MutableMapping) and "scores" in outputs:
            logits = outputs["scores"]
        else:
            logits = outputs
        logits = logits.contiguous().view(-1, logits.size(-1))

        if loss_name is None:
            raise ValueError(
                f"Transformer head {head_name} must either \
                                define a 'loss' in its config or return \
                                a dict that contains key 'losses'."
            )
        output = self.losses[loss_name](sample_list, {"scores": logits})
        return {"losses": output, "scores": logits}
