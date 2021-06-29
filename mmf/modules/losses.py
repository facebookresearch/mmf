# Copyright (c) Facebook, Inc. and its affiliates.
"""
Losses module contains implementations for various losses used generally
in vision and language space. One can register custom losses to be detected by
MMF using the following example.

.. code::

   from mmf.common.registry import registry
   from torch import nn


   @registry.register_loss("custom")
   class CustomLoss(nn.Module):
       ...

Then in your model's config you can specify ``losses`` attribute to use this loss
in the following way:

.. code::

   model_config:
       some_model:
           losses:
               - type: custom
               - params: {}
"""
import collections
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.utils.distributed import gather_tensor_along_batch_with_backward, get_rank
from mmf.utils.logger import log_class_usage
from omegaconf import MISSING
from packaging import version
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class LossConfig:
    type: str = MISSING
    params: Dict[str, Any] = MISSING


class Losses(nn.Module):
    """``Losses`` acts as an abstraction for instantiating and calculating
    losses. ``BaseModel`` instantiates this class based on the `losses`
    attribute in the model's configuration `model_config`. ``loss_list``
    needs to be a list for each separate loss containing `type` and `params`
    attributes.

    Args:
        loss_list (ListConfig): Description of parameter `loss_list`.

    Example::

        # losses:
        # - type: logit_bce
        # Can also contain `params` to specify that particular loss's init params
        # - type: combined
        config = [{"type": "logit_bce"}, {"type": "combined"}]
        losses = Losses(config)

    .. note::

        Since, ``Losses`` is instantiated in the ``BaseModel``, normal end user
        mostly doesn't need to use this class.

    Attributes:
        losses: List containing instantiations of each loss
                                   passed in config
    """

    # TODO: Union types are not supported in OmegaConf.
    # Later investigate for a workaround.for
    def __init__(self, loss_list: List[Union[str, LossConfig]]):
        super().__init__()
        self.losses = nn.ModuleList()
        config = registry.get("config")
        self._evaluation_predict = False
        if config:
            self._evaluation_predict = config.get("evaluation", {}).get(
                "predict", False
            )

        for loss in loss_list:
            self.losses.append(MMFLoss(loss))

    def forward(self, sample_list: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        """Takes in the original ``SampleList`` returned from DataLoader
        and `model_output` returned from the model and returned a Dict containing
        loss for each of the losses in `losses`.

        Args:
            sample_list (SampleList): SampleList given be the dataloader.
            model_output (Dict): Dict returned from model as output.

        Returns:
            Dict: Dictionary containing loss value for each of the loss.

        """
        output = {}
        if "targets" not in sample_list:
            if not self._evaluation_predict:
                warnings.warn(
                    "Sample list has not field 'targets', are you "
                    "sure that your ImDB has labels? you may have "
                    "wanted to run with evaluation.predict=true"
                )
            return output

        for loss in self.losses:
            output.update(loss(sample_list, model_output))

        if not torch.jit.is_scripting():
            registry_loss_key = "{}.{}.{}".format(
                "losses", sample_list["dataset_name"], sample_list["dataset_type"]
            )
            # Register the losses to registry
            registry.register(registry_loss_key, output)

        return output


class MMFLoss(nn.Module):
    """Internal MMF helper and wrapper class for all Loss classes.
    It makes sure that the value returned from a Loss class is a dict and
    contain proper dataset type in keys, so that it is easy to figure out
    which one is the val loss and which one is train loss.

    For example: it will return ``{"val/vqa2/logit_bce": 27.4}``, in case
    `logit_bce` is used and SampleList is from `val` set of dataset `vqa2`.

    Args:
        params (type): Description of parameter `params`.

    .. note::

        Since, ``MMFLoss`` is used by the ``Losses`` class, end user
        doesn't need to worry about it.
    """

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}

        is_mapping = isinstance(params, collections.abc.MutableMapping)

        if is_mapping:
            if "type" not in params:
                raise ValueError(
                    "Parameters to loss must have 'type' field to"
                    "specify type of loss to instantiate"
                )
            else:
                loss_name = params["type"]
        else:
            assert isinstance(
                params, str
            ), "loss must be a string or dictionary with 'type' key"
            loss_name = params

        self.name = loss_name

        loss_class = registry.get_loss_class(loss_name)

        log_class_usage("Loss", loss_class)

        if loss_class is None:
            raise ValueError(f"No loss named {loss_name} is registered to registry")
        # Special case of multi as it requires an array
        if loss_name.startswith("multi"):
            assert is_mapping
            self.loss_criterion = loss_class(params)
        else:
            if is_mapping:
                loss_params = params.get("params", {})
            else:
                loss_params = {}
            self.loss_criterion = loss_class(**loss_params)

    def forward(self, sample_list: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        loss = self.loss_criterion(sample_list, model_output)

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, dtype=torch.float)

        if loss.dim() == 0:
            loss = loss.view(1)

        if not torch.jit.is_scripting():
            key = "{}/{}/{}".format(
                sample_list.dataset_type, sample_list.dataset_name, self.name
            )
        else:
            key = f"{self.name}"
        return {key: loss}


@registry.register_loss("logit_bce")
class LogitBinaryCrossEntropy(nn.Module):
    """Returns Binary Cross Entropy for logits.

    Attention:
        `Key`: logit_bce
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy for logits

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="mean")

        return loss * targets.size(1)


@registry.register_loss("triple_logit_bce")
class TripleLogitBinaryCrossEntropy(nn.Module):
    """
    This is used for Three-branch fusion only. We predict scores and compute
    cross entropy loss for each of branches.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy for logits
        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.
        Returns:
            torch.FloatTensor: Float value for loss.
        """
        scores = model_output["scores"]
        targets = sample_list["targets"]

        if scores.dim() == 3:
            loss = (
                F.binary_cross_entropy_with_logits(
                    scores[:, 0], targets, reduction="mean"
                )
                + F.binary_cross_entropy_with_logits(
                    scores[:, 1], targets, reduction="mean"
                )
                + F.binary_cross_entropy_with_logits(
                    scores[:, 2], targets, reduction="mean"
                )
            )
        else:
            loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="mean")

        return loss * targets.size(-1)


@registry.register_loss("bce")
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]

        loss = F.binary_cross_entropy(scores, targets, reduction="mean")

        return loss * targets.size(1)


@registry.register_loss("caption_cross_entropy")
class CaptionCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the cross entropy loss for captions.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]

        # If no captions(test dataset) then assume decode length to be uniform
        if hasattr(sample_list, "caption_len"):
            caption_lengths, _ = sample_list.caption_len.sort(dim=0, descending=True)
            decode_lengths = (caption_lengths - 1).tolist()
        else:
            decode_lengths = [targets.size(1)] * targets.size(0)
        if version.parse(torch.__version__) >= version.parse("1.1"):
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            ).data
        else:
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = F.cross_entropy(scores, targets)

        return loss


@registry.register_loss("nll_loss")
class NLLLoss(nn.Module):
    """Negative log likelikehood loss."""

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the negative log likelihood.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        _, idx = targets.max(dim=1)
        loss = F.nll_loss(scores, idx, reduction="mean")

        return loss * targets.size(1)


def kl_div(log_x, y):
    y_is_0 = torch.eq(y.data, 0)
    y.data.masked_fill_(y_is_0, 1)
    log_y = torch.log(y)
    y.data.masked_fill_(y_is_0, 0)
    res = y * (log_y - log_x)

    return torch.sum(res, dim=1, keepdim=True)


@registry.register_loss("multi")
class MultiLoss(nn.Module):
    """A loss for combining multiple losses with weights.

    Args:
        params (List(Dict)): A list containing parameters for each different loss
                             and their weights.

    Example::

        # MultiLoss works with config like below where each loss's params and
        # weights are defined
        losses:
        - type: multi
          params:
          - type: logit_bce
            weight: 0.3
            params: {}
          - type: attention_supervision
            weight: 0.7
            params: {}

    """

    def __init__(self, params):
        super().__init__()
        self.losses = []
        self.losses_weights = []

        self.loss_names = []

        for loss_params in params["params"]:
            self.loss_names.append(loss_params["type"])
            loss_fn = MMFLoss(loss_params)
            loss_weight = loss_params.get("weight", {})
            self.losses.append(loss_fn)
            self.losses_weights.append(loss_weight)

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Calculates and returns the multi loss.

        Args:
            sample_list (SampleList): SampleList containing `attentions` attribute.
            model_output (Dict): Model output containing `attention_supervision`
                                 attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        loss = 0
        for idx, loss_fn in enumerate(self.losses):
            value = loss_fn(sample_list, model_output, *args, **kwargs)
            loss += self.losses_weights[idx] * list(value.values())[0]
        return loss


@registry.register_loss("attention_supervision")
class AttentionSupervisionLoss(nn.Module):
    """Loss for attention supervision. Used in case you want to make attentions
    similar to some particular values.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = lambda *args, **kwargs: nn.functional.binary_cross_entropy(
            *args, **kwargs
        )

    def forward(self, sample_list, model_output):
        """Calculates and returns the multi loss.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        context_attentions = model_output["attentions"]
        attention_supervision = sample_list["info"]["attention_supervision"]

        loss = self.loss_fn(
            context_attentions[0],
            attention_supervision.float(),
            weight=attention_supervision.float(),
        )

        # Multiply average loss back with target size to get actual loss
        return loss * attention_supervision.size(1)


@registry.register_loss("weighted_softmax")
class WeightedSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = loss * tar_sum
        loss = torch.sum(loss) / loss.size(0)
        return loss


@registry.register_loss("softmax_kldiv")
class SoftmaxKlDivLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = torch.sum(loss) / loss.size(0)
        return loss


@registry.register_loss("wrong")
class WrongLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = F.kl_div(res, tar, reduction="mean")
        loss *= target_score.size(1)
        return loss


@registry.register_loss("bce_kl_combined")
class CombinedLoss(nn.Module):
    def __init__(self, weight_softmax):
        super().__init__()
        self.weight_softmax = weight_softmax

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss1 = kl_div(res, tar)
        loss1 = torch.sum(loss1) / loss1.size(0)

        loss2 = F.binary_cross_entropy_with_logits(
            pred_score, target_score, reduction="mean"
        )
        loss2 *= target_score.size(1)

        loss = self.weight_softmax * loss1 + loss2

        return loss


@registry.register_loss("m4c_decoding_bce_with_mask")
class M4CDecodingBCEWithMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.0])

    def forward(self, sample_list, model_output):
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss_mask = sample_list["train_loss_mask"]
        assert scores.dim() == 3 and loss_mask.dim() == 2

        losses = F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
        losses *= loss_mask.unsqueeze(-1)

        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count
        return loss


@registry.register_loss("cross_entropy")
class CrossEntropyLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**params)

    def forward(self, sample_list, model_output):
        return self.loss_fn(model_output["scores"], sample_list["targets"])


@registry.register_loss("soft_label_cross_entropy")
class SoftLabelCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="mean", normalize_targets=True):
        assert reduction in (
            "mean",
            "sum",
        ), "Argument `reduction` only supports `mean` and `sum`"

        super().__init__()

        self.ignore_index = ignore_index
        self.reduction = reduction
        self.normalize_targets = normalize_targets
        self.eps = torch.finfo(torch.float32).eps

    @staticmethod
    def convert_to_one_hot(targets, n_classes):
        one_hot_targets = torch.zeros(
            (targets.size(0), n_classes), dtype=torch.long, device=targets.device
        )
        one_hot_targets.scatter_(1, targets.long().view(-1, 1), 1)
        return one_hot_targets

    def compute_loss(self, targets, scores):
        """for N examples and C classes
        - scores: N x C these are raw outputs (without softmax/sigmoid)
        - targets: N x C or N corresponding targets

        Target elements set to ignore_index contribute 0 loss.

        Samples where all entries are ignore_index do not contribute to the loss
        reduction.
        """

        assert targets.size(0) == scores.size(
            0
        ), "`targets` and `scores` should have the same batch size"

        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
            mask = targets.ne(self.ignore_index).float()  # mask out `ignore_index`
        else:
            mask = targets.sum(-1, keepdim=True).ne(0).float()  # mask out zero rows

        if targets.size(1) == 1:
            targets = self.convert_to_one_hot(targets, scores.size(1))
        targets = targets.float() * mask

        if self.normalize_targets:
            targets /= self.eps + targets.sum(dim=1, keepdim=True)

        per_sample_per_target_loss = -targets * F.log_softmax(scores, dim=-1)
        per_sample_loss = torch.sum(per_sample_per_target_loss, -1)
        loss = per_sample_loss.sum()
        # perform reduction
        if self.reduction == "mean":
            # normalize based on the number of samples with > 0 non-ignored targets
            loss /= torch.sum(torch.sum(mask, -1) > 0).clamp(min=1)
        return loss

    def forward(self, sample_list, model_output):
        return self.compute_loss(sample_list["targets"], model_output["scores"])


@registry.register_loss("label_smoothing_cross_entropy")
class LabelSmoothingCrossEntropyLoss(SoftLabelCrossEntropyLoss):
    """Cross-entropy loss with label smoothing. If `label_smoothing` = 0, then
    it's canonical cross entropy.
    The smoothed one-hot encoding is 1 - label_smoothing for true label and
    label_smoothing / (num_classes - 1) for the rest.

    Reference: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
    """

    def __init__(self, label_smoothing=0.1, reduction="mean", ignore_index=-100):
        assert (
            0 <= label_smoothing < 1
        ), "value of argument `label_smoothing` must be in range [0, 1)."

        super().__init__(ignore_index, reduction, False)
        self.label_smoothing = label_smoothing

    def smooth_targets(self, targets, n_classes):
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        mask = targets.ne(self.ignore_index)

        smoothing_value = self.label_smoothing / (n_classes - 1)
        one_hot = torch.full(
            (n_classes,), smoothing_value, device=targets.device
        ).repeat(targets.size(0), 1)
        # mask out target with `ignore_index` to avoid error `index out of bounds`
        one_hot.scatter_(1, targets * mask.long(), 1 - self.label_smoothing)
        return one_hot * mask.float()

    def forward(self, sample_list, model_output):
        scores = model_output["scores"]
        one_hot = self.smooth_targets(sample_list["targets"], scores.size(1))
        loss = self.compute_loss(one_hot, scores)
        return loss


@registry.register_loss("in_batch_hinge")
class InBatchHinge(nn.Module):
    """
    Based on the code from https://github.com/fartashf/vsepp/blob/master/model.py
    """

    def __init__(self, margin: float = 0.0, hard: bool = False):
        super().__init__()
        self.margin = margin
        self.hard = hard

    def _compute_loss(self, correlations: Tensor):
        diagonal = correlations.diag()[:, None]
        d1 = diagonal.expand_as(correlations)
        d2 = diagonal.t().expand_as(correlations)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + correlations - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + correlations - d2).clamp(min=0)

        # clear diagonals
        mask = 1 - torch.eye(correlations.size(0), device=correlations.device)
        cost_s = cost_s * mask
        cost_im = cost_im * mask

        if self.hard:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

    def forward(self, sample_list: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        image_embeddings = model_output["scores"]
        text_embeddings = model_output["targets"]

        if image_embeddings.shape[0] == text_embeddings.shape[0]:
            # Training/Single-GT loss
            correlations = image_embeddings @ text_embeddings.t()
            loss = self._compute_loss(correlations)
        else:
            # Evaluation/Multi-GT loss
            assert text_embeddings.shape[0] % image_embeddings.shape[0] == 0

            batch_size, dim_size = image_embeddings.shape
            factor = text_embeddings.shape[0] // image_embeddings.shape[0]
            text_embeddings = text_embeddings.reshape(batch_size, factor, dim_size)
            correlations = image_embeddings @ text_embeddings.permute(1, 2, 0)  # FxBxB

            loss = 0
            for corr in correlations:
                loss += self._compute_loss(corr)

        return loss


@registry.register_loss("contrastive_loss")
class ContrastiveLoss(nn.Module):
    """
    This is a generic contrastive loss typically used for pretraining. No modality
    assumptions are made here.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        assert (
            "embedding_1" in model_output and "embedding_2" in model_output
        ), "Embedding names must be available before loss calculation"

        embedding_1 = model_output["embedding_1"]
        embedding_2 = model_output["embedding_2"]

        assert embedding_1.size(0) == embedding_2.size(0), "batch size must match"
        per_gpu_batch_size = embedding_1.size(0)

        embedding_1_all_gpus = gather_tensor_along_batch_with_backward(embedding_1)
        embedding_2_all_gpus = gather_tensor_along_batch_with_backward(embedding_2)

        temperature = model_output["temperature"]

        logits_1 = torch.matmul(embedding_1, embedding_2_all_gpus.transpose(0, 1))
        logits_2 = torch.matmul(embedding_2, embedding_1_all_gpus.transpose(0, 1))
        labels = per_gpu_batch_size * get_rank() + torch.arange(
            per_gpu_batch_size, device=temperature.device
        )

        loss_1 = F.cross_entropy(logits_1, labels)
        loss_2 = F.cross_entropy(logits_2, labels)

        return (loss_1 + loss_2) / 2


@registry.register_loss("mse")
class MSELoss(nn.Module):
    """Mean Squared Error loss"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, sample_list, model_output):
        targets = sample_list["targets"]
        scores = model_output["scores"]
        loss = self.loss_fn(scores, targets)
        return loss


@registry.register_loss("cos_emb_loss")
class CosineEmbeddingLoss(nn.Module):
    """Cosine embedding loss"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(self, sample_list, model_output):
        targets = sample_list["targets"]
        scores = model_output["scores"]
        y = torch.ones(targets.size(0)).to(targets.device)
        loss = self.loss_fn(scores, targets, y)
        return loss
