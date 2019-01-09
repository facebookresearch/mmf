# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
import torch.nn.functional as F

from pythia.core.registry import registry


class Loss(nn.Module):
    def __init__(self, params={}):
        super(Loss, self).__init__()
        if type(params) == str:
            loss_type = params
        else:
            loss_type = params['type']
        if loss_type == 'logit_bce':
            self.loss_criterion = LogitBinaryCrossEntropy()
        elif loss_type == 'softmax_kl':
            self.loss_criterion = SoftmaxKlDivLoss()
        elif loss_type == 'wrong':
            self.loss_criterion = WrongLoss()
        elif loss_type == 'combined':
            self.loss_criterion = CombinedLoss()
        elif loss_type == 'mse':
            self.loss_criterion = nn.MSELoss()
        elif loss_type == 'bce':
            self.loss_criterion = BinaryCrossEntropyLoss()
        elif loss_type == 'nll':
            self.loss_criterion = NLLLoss()
        elif loss_type == 'attention_supervision':
            self.loss_criterion = AttentionSupervisionLoss()
        elif loss_type == 'multi':
            self.loss_criterion = MultiLoss(params['params'])
        elif hasattr(nn, loss_type):
            self.loss_criterion = getattr(nn, loss_type)()
        else:
            raise NotImplementedError("Unknown loss type: %s" % loss_type)

    def forward(self, *args, **kwargs):
        return self.loss_criterion(*args, **kwargs)


class LogitBinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(LogitBinaryCrossEntropy, self).__init__()

    def forward(self, pred_score, target_score, info={}, weights=None):
        loss = F.binary_cross_entropy_with_logits(pred_score,
                                                  target_score,
                                                  reduction='mean')

        return loss * target_score.size(1)


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, pred_score, target_score, info={}, weights=None):
        loss = F.binary_cross_entropy(pred_score, target_score,
                                      reduction='mean')

        return loss * target_score.size(1)


class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, pred_score, target_score, info={}, weights=None):
        _, idx = target_score.max(dim=1)
        loss = F.nll_loss(pred_score, idx,
                          reduction='mean')

        return loss * target_score.size(1)


def kl_div(log_x, y):
    y_is_0 = torch.eq(y.data, 0)
    y.data.masked_fill_(y_is_0, 1)
    log_y = torch.log(y)
    y.data.masked_fill_(y_is_0, 0)
    res = y * (log_y - log_x)

    return torch.sum(res, dim=1, keepdim=True)


class MultiLoss(nn.Module):
    def __init__(self, params):
        super(MultiLoss, self).__init__()
        self.losses = []
        self.losses_weights = []
        self.writer = registry.get('writer')

        self.loss_names = []

        for loss_params in params:
            self.loss_names.append(loss_params['type'])
            loss_fn = Loss(loss_params)
            loss_weight = loss_params.get('weight', {})
            self.losses.append(loss_fn)
            self.losses_weights.append(loss_weight)

    def forward(self, pred_score, target_score, info={}):
        loss = 0
        iteration = registry.get('current_iteration')

        for idx, loss_fn in enumerate(self.losses):
            value = loss_fn(pred_score, target_score, info)
            self.writer.add_scalar(self.loss_names[idx], value, iteration)
            loss += self.losses_weights[idx] * value

        return loss


class AttentionSupervisionLoss(nn.Module):
    def __init__(self):
        super(AttentionSupervisionLoss, self).__init__()
        self.loss_fn = lambda *args, **kwargs: \
            nn.functional.binary_cross_entropy(*args, **kwargs)

    def forward(self, pred_score, target_score, info):
        # TODO: Create this an option so that this becomes zero
        # when att sup is not passed. As in, don't pass in att sup
        batch = info['batch']
        attention_supervision = batch['info']['attention_supervision']
        context_attentions = info['context_attentions']

        loss = self.loss_fn(context_attentions[0],
                            attention_supervision.float(),
                            weight=attention_supervision.float())

        # Multiply average loss back with target size to get actual loss
        return loss * attention_supervision.size(1)


class WeightedSoftmaxLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftmaxLoss, self).__init__()

    def forward(self, pred_score, target_score, info={}):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = loss * tar_sum
        loss = torch.sum(loss) / loss.size(0)
        return loss


class SoftmaxKlDivLoss(nn.Module):
    def __init__(self):
        super(SoftmaxKlDivLoss, self).__init__()

    def forward(self, pred_score, target_score, info={}):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = torch.sum(loss) / loss.size(0)
        return loss


class WrongLoss(nn.Module):
    def __init__(self):
        super(WrongLoss, self).__init__()

    def forward(self, pred_score, target_score, info={}):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = F.kl_div(res, tar, reduction='mean')
        loss *= target_score.size(1)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, weight_softmax):
        super(CombinedLoss, self).__init__()
        self.weight_softmax = weight_softmax

    def forward(self, pred_score, target_score, info={}):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss1 = kl_div(res, tar)
        loss1 = torch.sum(loss1) / loss1.size(0)

        loss2 = F.binary_cross_entropy_with_logits(pred_score,
                                                   target_score,
                                                   reduction='mean'
                                                   )
        loss2 *= target_score.size(1)

        loss = self.weight_softmax * loss1 + loss2

        return loss
