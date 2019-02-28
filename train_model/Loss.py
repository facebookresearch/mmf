# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
import torch.nn.functional as F

""" Losses are enclosed within nn.Module sub-classes.

    Parameters
    ----------
    pred_score: Tensor of size [N,K] with logits
    target_score: Tensor of size [N,K] with target scores

    With N samples in each batch and K label categories

    Returns
    ----------
    loss: Single valued tensor, normalized only across the batch.
"""


def get_loss_criterion(loss_list):
    """Returns a list of losses to be used.

        Parameter
        ----------
        loss_list: List of the losses supported. Max length is 2.
    """
    loss_criterions = []
    if len(loss_list) == 2:
        if loss_list[0] == 'softmaxKL':
            print('Training with Complement Objective only supports softmaxKL'
                  ' as the primary loss. Current primary loss is: ',
                  loss_list[0])
            raise NotImplementedError
    elif len(loss_list) > 2:
        raise NotImplementedError

    for loss_config in loss_list:
        if loss_config == 'logitBCE':
            loss_criterion = LogitBinaryCrossEntropy()
        elif loss_config == 'softmaxKL':
            loss_criterion = SoftmaxKlDivLoss()
        elif loss_config == 'wrong':
            loss_criterion = wrong_loss()
        elif loss_config == 'combined':
            loss_criterion = CombinedLoss()
        elif loss_config == 'complementEntropy':
            loss_criterion = ComplementEntropyLoss()
        else:
            raise NotImplementedError
        loss_criterions.append(loss_criterion)
    return loss_criterions


class LogitBinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(LogitBinaryCrossEntropy, self).__init__()

    def forward(self, pred_score, target_score, weights=None):
        loss = F.binary_cross_entropy_with_logits(pred_score,
                                                  target_score,
                                                  size_average=True)
        loss = loss * target_score.size(1)
        return loss


def kl_div(log_x, y):
    y_is_0 = torch.eq(y.data, 0)
    y.data.masked_fill_(y_is_0, 1)
    log_y = torch.log(y)
    y.data.masked_fill_(y_is_0, 0)
    res = y * (log_y - log_x)

    return torch.sum(res, dim=1, keepdim=True)


def complement_entropy_loss(x, y):
    """ Returns the complement entropy loss as proposed in the report.

        This implementation is faithful with Equation (6) in the report's
        section (2.4).

        Equation (6) talks about the complement entropy, we calculate the
        negative of its value i.e. complement entropy loss.
    """
    # --------------------------------------------------------------------------
    # Negated complement entropy (loss) for each label with zero target score
    # --------------------------------------------------------------------------
    y_is_0 = torch.eq(y.data, 0)
    x_remove_0 = x.clone().data.masked_fill_(y_is_0, 0)
    xr_sum = torch.sum(x_remove_0, dim=1, keepdim=True)
    one_min_xr_sum = 1-xr_sum
    one_min_xr_sum.masked_fill_(one_min_xr_sum <= 0, 1e-7)  # Numerical issues
    px = x / one_min_xr_sum
    log_px = torch.log(px + 1e-10)  # Numerical issues
    new_x = px * log_px
    loss = new_x * (y_is_0.float())  # Remove non-zero labels loss

    # --------------------------------------------------------------------------
    # Normalize the loss to balance it with cross entropy loss
    # --------------------------------------------------------------------------
    num_labels = y.size()[1]
    zero_labels = torch.sum(y_is_0, dim=1, keepdim=True).float()
    non_zero_labels = num_labels - zero_labels
    zero_labels.masked_fill_(torch.eq(zero_labels.data, 0), 1e-7)  # num. issues
    normalize = non_zero_labels / zero_labels
    zero_labels.masked_fill_(torch.eq(zero_labels.data, 0), 0)
    loss = loss * normalize
    return torch.sum(loss, dim=1, keepdim=True)  # Sum the loss over the labels


class weighted_softmax_loss(nn.Module):
    def __init__(self):
        super(weighted_softmax_loss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = loss * tar_sum
        loss = torch.sum(loss) / loss.size(0)
        return loss


class ComplementEntropyLoss(nn.Module):
    """ Complement Entropy that maximizes entropy of non-ground truth
        labels. It was proposed to complement the classification loss.

        Paper Link : https://openreview.net/pdf?id=HyM7AiA5YX

        The same approach can be extended to multi-label classification problems
        with Softmax KL divergence loss.

        Report Link :
        https://drive.google.com/file/d/16NtLvZvBPq1cRVeCSq7sXXg0C8NkSi4l/view

        This implementation is faithful with Equation (6) in the report's
        section (2.4).

        All the target scores with non-zero values are treated as positive
        labels while calculating the complement entropy.

        When using Softmax KL divergence loss, predictions corresponding to
        incorrect labels do not directly contribute to the training
        (parameter updates).

        This complementary loss could be used to add an explicit objective for
        maximizing the entropy of the incorrect labels.

        While training, we alternate between the primary and the complement
        objective.
    """

    def __init__(self):
        super(ComplementEntropyLoss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum
        res = F.softmax(pred_score, dim=1)
        loss = complement_entropy_loss(res, tar)
        loss = torch.sum(loss) / loss.size(0)
        return loss


class SoftmaxKlDivLoss(nn.Module):
    def __init__(self):
        super(SoftmaxKlDivLoss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = torch.sum(loss) / loss.size(0)
        return loss


class wrong_loss(nn.Module):
    def __init__(self):
        super(wrong_loss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = F.kl_div(res, tar, size_average=True)
        loss *= target_score.size(1)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, weight_softmax, weight_complement=None):
        super(CombinedLoss, self).__init__()
        self.weight_softmax = weight_softmax
        self.weight_complement = weight_complement

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss1 = kl_div(res, tar)
        loss1 = torch.sum(loss1) / loss1.size(0)

        loss2 = F.binary_cross_entropy_with_logits(pred_score,
                                                   target_score,
                                                   size_average=True)
        loss2 *= target_score.size(1)

        loss = self.weight_softmax * loss1 + loss2

        # ----------------------------------------------------------------------
        # Combine complement entropy loss pre-multiplied with a weight
        # ----------------------------------------------------------------------
        if self.weight_complement is not None:
            res = F.softmax(pred_score, dim=1)
            loss3 = complement_entropy_loss(res, tar)
            loss3 = torch.sum(loss3) / loss3.size(0)
            loss += self.weight_complement * loss3

        return loss
