import torch
import torch.nn as nn
import torch.nn.functional as F

from pythia.common.registry import registry


class Losses(nn.Module):
    def __init__(self, loss_list):
        super().__init__()
        self.losses = []
        for loss in loss_list:
            self.losses.append(Loss(loss))

    def forward(self, sample_list, model_output, *args, **kwargs):
        output = {}

        if not hasattr(sample_list, "targets"):
            return output

        for loss in self.losses:
            output.update(loss(sample_list, model_output, *args, **kwargs))

        return output

class Loss(nn.Module):
    def __init__(self, params={}):
        super().__init__()
        self.writer = registry.get("writer")
        if "type" not in params:
            raise ValueError("Parameters to loss must have 'type' field to"
                             "specify type of loss to instantiate")

        loss_name = params["type"]
        self.name = loss_name

        loss_class = registry.get_loss_class(loss_name)

        if loss_class is None:
            raise ValueError("No loss named {} is registered to registry"
                             .format(loss_name))
        # Special case of multi as it requires an array
        if loss_name == "multi":
            self.loss_criterion = loss_class(params)
        else:
            loss_params = params.get("params", {})
            self.loss_criterion = loss_class(**loss_params)

    def forward(self, sample_list, model_output, *args, **kwargs):
        loss = self.loss_criterion(sample_list, model_output, *args, **kwargs)

        if loss.dim() == 0:
            loss = loss.view(1)
        return {
            "{}/{}".format(sample_list.dataset_type, self.name): loss
        }


@registry.register_loss("logit_bce")
class LogitBinaryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss = F.binary_cross_entropy_with_logits(scores, targets,
                                                  reduction="mean")

        return loss * targets.size(1)


@registry.register_loss("bce")
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss = F.binary_cross_entropy(scores, targets,
                                      reduction="mean")

        return loss * targets.size(1)


@registry.register_loss("nll_loss")
class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
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
    def __init__(self, params):
        super().__init__()
        self.losses = []
        self.losses_weights = []
        self.writer = registry.get('writer')

        self.loss_names = []

        for loss_params in params["params"]:
            self.loss_names.append(loss_params['type'])
            loss_fn = Loss(loss_params)
            loss_weight = loss_params.get('weight', {})
            self.losses.append(loss_fn)
            self.losses_weights.append(loss_weight)

    def forward(self, sample_list, model_output, *args, **kwargs):
        loss = 0
        iteration = registry.get('current_iteration')

        for idx, loss_fn in enumerate(self.losses):
            value = loss_fn(sample_list, model_output, *args, **kwargs)
            self.writer.add_scalar(self.loss_names[idx], value, iteration)
            loss += self.losses_weights[idx] * value

        return loss


@registry.register_loss("attention_supervision")
class AttentionSupervisionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = lambda *args, **kwargs: \
            nn.functional.binary_cross_entropy(*args, **kwargs)

    def forward(self, sample_list, model_output):
        context_attentions = model_output["context_attentions"]
        attention_supervision = sample_list["info"]["attention_supervision"]

        loss = self.loss_fn(context_attentions[0],
                            attention_supervision.float(),
                            weight=attention_supervision.float())

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

        loss2 = F.binary_cross_entropy_with_logits(pred_score,
                                                   target_score,
                                                   reduction="mean")
        loss2 *= target_score.size(1)

        loss = self.weight_softmax * loss1 + loss2

        return loss
