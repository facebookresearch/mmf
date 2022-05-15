# Copyright (c) Facebook, Inc. and its affiliates.

# Mostly copy-pasted from
# https://github.com/facebookresearch/detr/blob/master/models/detr.py
from typing import Optional

import torch
import torch.nn.functional as F
from mmf.models.unit.backbone import build_unit_convnet_backbone
from mmf.models.unit.matcher import HungarianMatcher
from mmf.models.unit.misc import NestedTensor
from mmf.models.unit.transformer import UniTTransformer
from mmf.utils import box_ops
from mmf.utils.distributed import get_world_size, is_dist_initialized
from torch import nn, Tensor


class UniTBaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_queries = args.num_queries
        self.backbone = build_unit_convnet_backbone(args)
        self.transformer = UniTTransformer(args)
        encoder_hidden_dim = self.transformer.d_model_enc
        decoder_hidden_dim = self.transformer.d_model_dec

        self.query_embeds = nn.ModuleDict()
        for task_type in self.num_queries:
            task_dict = nn.ModuleDict()
            for dataset in self.num_queries[task_type]:
                task_dict[dataset] = nn.Embedding(
                    self.num_queries[task_type][dataset], decoder_hidden_dim
                )
            self.query_embeds[task_type] = task_dict

        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, encoder_hidden_dim, kernel_size=1
        )

    def forward(
        self,
        img_src: Tensor,
        text_src: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        text_pos: Optional[Tensor] = None,
        output_hidden_states_only: bool = False,
        task_type: str = "detection",
        dataset_name: str = "detection_coco",
        task_idx: Optional[int] = None,
    ):
        img_mask = None
        img_pos = [None]
        if img_src is not None:
            if not isinstance(img_src, NestedTensor):
                img_src = NestedTensor.from_tensor_list(img_src)
            features, img_pos = self.backbone(img_src)

            img_src, img_mask = features[-1].decompose()
            img_src = self.input_proj(img_src)

        query_embed = self.query_embeds[task_type][dataset_name]
        hs, _ = self.transformer(
            img_src=img_src,
            img_mask=img_mask,
            img_pos=img_pos[-1],
            text_src=text_src,
            text_mask=text_mask,
            text_pos=text_pos,
            query_embed=query_embed.weight,
            task_type=task_type,
            dataset_name=dataset_name,
            task_idx=task_idx,
        )

        if hs is not None:
            assert hs.size(2) == self.num_queries[task_type][dataset_name]

        return {"hidden_states": hs}


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class AttributeHead(nn.Module):
    def __init__(self, object_class_num, attribute_class_num, representation_size):
        super().__init__()

        # copy-pasted from
        # https://github.com/ronghanghu/vqa-maskrcnn-benchmark-m4c/blob/0fbccee2dfed10d652bcf014f9f8bfafd8478f51/maskrcnn_benchmark/modeling/roi_heads/box_head/roi_box_predictors.py#L52-L61  # NoQA
        self.cls_embed = nn.Embedding(object_class_num + 1, 256)
        self.attr_linear1 = nn.Linear(representation_size + 256, 512)
        self.attr_linear2 = nn.Linear(512, attribute_class_num)

        nn.init.normal_(self.cls_embed.weight, std=0.01)
        nn.init.normal_(self.attr_linear1.weight, std=0.01)
        nn.init.normal_(self.attr_linear2.weight, std=0.01)
        nn.init.constant_(self.attr_linear1.bias, 0)
        nn.init.constant_(self.attr_linear2.bias, 0)

    def forward(self, hidden_states: Tensor, labels: Tensor):
        # copy-pasted from
        # https://github.com/ronghanghu/vqa-maskrcnn-benchmark-m4c/blob/0fbccee2dfed10d652bcf014f9f8bfafd8478f51/maskrcnn_benchmark/modeling/roi_heads/box_head/roi_box_predictors.py#L76-L96  # NoQA

        # get embeddings of indices using gt cls labels
        cls_embed_out = self.cls_embed(labels)

        # concat with fc7 feats
        concat_attr = torch.cat([hidden_states, cls_embed_out], dim=-1)

        # pass through attr head layers
        fc_attr = self.attr_linear1(concat_attr)
        attr_score = F.relu(self.attr_linear2(fc_attr))

        return attr_score


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs
           of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class
           and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        attribute_head=None,
        attribute_class_num=None,
        max_attribute_num=None,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object
                category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values
                their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of
                available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.attribute_head = attribute_head
        self.attribute_class_num = attribute_class_num
        self.max_attribute_num = max_attribute_num

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim
            [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        if self.attribute_head is not None and "attributes" in targets[0]:
            attribute_logits = self.attribute_head(
                outputs["hs_for_attr"], target_classes
            )
            target_attributes_o = torch.cat(
                [t["attributes"][J] for t, (_, J) in zip(targets, indices)]
            )
            target_attributes = -torch.ones(
                *src_logits.shape[:2], 16, dtype=torch.int64, device=src_logits.device
            )
            target_attributes[idx] = target_attributes_o
            losses["loss_attr"] = self._attribute_loss(
                attribute_logits, target_attributes
            )

        return losses

    def loss_labels_balanced(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim
            [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        sl = src_logits.flatten(0, 1)
        tc = target_classes.flatten(0, 1)
        pos = tc != self.num_classes
        loss_pos = F.cross_entropy(sl[pos], tc[pos], reduction="none").sum() / num_boxes
        loss_neg = F.cross_entropy(sl[~pos], tc[~pos], reduction="none").sum() / (
            sl.shape[0] - num_boxes
        )

        loss_ce = (1 - self.eos_coef) * loss_pos + self.eos_coef * loss_neg
        losses = {"loss_ce": loss_ce}

        if self.attribute_head is not None:
            raise NotImplementedError()

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of
        predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't
        propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last
        # class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and
             the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim
             [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w),
             normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes).float(),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "labels_balanced": self.loss_labels_balanced,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for
                      the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see
                      each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for
        # normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each
        # intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss in ("labels", "labels_balanced"):
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def _attribute_loss(self, attribute_logits, attributes):
        _N, _B, _C = attribute_logits.size()
        assert _C == self.attribute_class_num
        attribute_logits = attribute_logits.view(_N * _B, _C)

        assert attributes.size(0) == _N
        assert attributes.size(1) == _B
        assert attributes.size(2) == self.max_attribute_num
        attributes = attributes.view(_N * _B, self.max_attribute_num)

        # https://github.com/ronghanghu/vqa-maskrcnn-benchmark-m4c/blob/0fbccee2dfed10d652bcf014f9f8bfafd8478f51/maskrcnn_benchmark/modeling/roi_heads/box_head/loss.py#L185-L222  # NoQA
        n_boxes = attribute_logits.shape[0]

        # N_BOXES, N_ATTR -> N_BOXES, 1, N_ATTR
        attribute_logits = attribute_logits.unsqueeze(1)

        # N_BOXES, 1, N_ATTR -> N_BOXES, MAX_ATTR_PER_INST, N_ATTR
        # -> N_BOXES * MAX_ATTR_PER_INST, N_ATTR
        attribute_logits = (
            attribute_logits.expand(n_boxes, 16, self.attribute_class_num)
            .contiguous()
            .view(-1, self.attribute_class_num)
        )

        # Normalize each box loss by # of valid GT attributes (ie attr != -1)
        # Repeat number of valid attributes per box along the rows and take transpose
        inv_per_box_weights = (
            (attributes >= 0).sum(dim=1).repeat(16, 1).transpose(0, 1).flatten()
        )
        per_box_weights = inv_per_box_weights.float().reciprocal()
        per_box_weights[per_box_weights > 1] = 0.0

        attributes = attributes.view(-1)
        attribute_loss = 0.5 * F.cross_entropy(
            attribute_logits, attributes, reduction="none", ignore_index=-1
        )

        attribute_loss = (attribute_loss * per_box_weights).view(n_boxes, -1).sum(dim=1)

        # Find number of boxes with atleast valid attribute
        n_valid_boxes = len(attribute_loss.nonzero())

        if n_valid_boxes > 0:
            attribute_loss = (attribute_loss / n_valid_boxes).sum()
        else:
            attribute_loss = (attribute_loss * 0.0).sum()

        return attribute_loss


def build_detection_loss(args, num_classes, attribute_head):
    matcher = HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        logsoftmax=args.use_bcl,
    )
    weight_dict = {"loss_ce": 1, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    weight_dict["loss_attr"] = args.attr_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = []
    if args.use_bcl:
        losses.append("labels_balanced")
    else:
        losses.append("labels")
    losses.extend(["boxes", "cardinality"])

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        attribute_head=attribute_head,
        attribute_class_num=args.attribute_class_num,
        max_attribute_num=args.max_attribute_num,
    )
    return criterion
