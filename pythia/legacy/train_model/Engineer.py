# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from config.config import cfg
from global_variables.global_variables import use_cuda
from tools.timer import Timer


def masked_unk_softmax(x, dim, mask_idx):
    x1 = F.softmax(x, dim=dim)
    x1[:, mask_idx] = 0
    x1_sum = torch.sum(x1, dim=1, keepdim=True)
    y = x1 / x1_sum
    return y


def compute_score_with_logits(logits, labels):
    logits = masked_unk_softmax(logits, 1, 0)
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size())
    one_hots = one_hots.cuda() if use_cuda else one_hots
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


def clip_gradients(myModel, i_iter, writer):
    max_grad_l2_norm = cfg.training_parameters.max_grad_l2_norm
    clip_norm_mode = cfg.training_parameters.clip_norm_mode
    if max_grad_l2_norm is not None:
        if clip_norm_mode == "all":
            norm = nn.utils.clip_grad_norm(myModel.parameters(), max_grad_l2_norm)
            writer.add_scalar("grad_norm", norm, i_iter)
        elif clip_norm_mode == "question":
            norm = nn.utils.clip_grad_norm(
                myModel.module.question_embedding_models.parameters(), max_grad_l2_norm
            )
            writer.add_scalar("question_grad_norm", norm, i_iter)
        else:
            raise NotImplementedError


def save_a_report(
    i_iter,
    train_loss,
    train_acc,
    train_avg_acc,
    report_timer,
    writer,
    data_reader_eval,
    myModel,
    loss_criterion,
):
    val_batch = next(iter(data_reader_eval))
    val_score, val_loss, n_val_sample = compute_a_batch(
        val_batch, myModel, eval_mode=True, loss_criterion=loss_criterion
    )
    val_acc = val_score / n_val_sample

    print(
        "iter:",
        i_iter,
        "train_loss: %.4f" % train_loss,
        " train_score: %.4f" % train_acc,
        " avg_train_score: %.4f" % train_avg_acc,
        "val_score: %.4f" % val_acc,
        "val_loss: %.4f" % val_loss.data[0],
        "time(s): % s" % report_timer.end(),
    )
    sys.stdout.flush()
    report_timer.start()

    writer.add_scalar("train_loss", train_loss, i_iter)
    writer.add_scalar("train_score", train_acc, i_iter)
    writer.add_scalar("train_score_avg", train_avg_acc, i_iter)
    writer.add_scalar("val_score", val_score, i_iter)
    writer.add_scalar("val_loss", val_loss.data[0], i_iter)
    for name, param in myModel.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), i_iter)


def save_a_snapshot(
    snapshot_dir,
    i_iter,
    iepoch,
    myModel,
    my_optimizer,
    loss_criterion,
    best_val_accuracy,
    best_epoch,
    best_iter,
    snapshot_timer,
    data_reader_eval,
):
    model_snapshot_file = os.path.join(snapshot_dir, "model_%08d.pth" % i_iter)
    model_result_file = os.path.join(snapshot_dir, "result_on_val.txt")
    save_dic = {
        "epoch": iepoch,
        "iter": i_iter,
        "state_dict": myModel.state_dict(),
        "optimizer": my_optimizer.state_dict(),
    }

    if data_reader_eval is not None:
        val_accuracy, avg_loss, val_sample_tot = one_stage_eval_model(
            data_reader_eval, myModel, loss_criterion=loss_criterion
        )
        print(
            "i_epoch:",
            iepoch,
            "i_iter:",
            i_iter,
            "val_loss:%.4f" % avg_loss,
            "val_acc:%.4f" % val_accuracy,
            "runtime: %s" % snapshot_timer.end(),
        )
        snapshot_timer.start()
        sys.stdout.flush()

        with open(model_result_file, "a") as fid:
            fid.write("%d %d %.5f\n" % (iepoch, i_iter, val_accuracy))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = iepoch
            best_iter = i_iter
            best_model_snapshot_file = os.path.join(snapshot_dir, "best_model.pth")

        save_dic["best_val_accuracy"] = best_val_accuracy
        torch.save(save_dic, model_snapshot_file)

        if best_iter == i_iter:
            if os.path.exists(best_model_snapshot_file):
                os.remove(best_model_snapshot_file)
            os.link(model_snapshot_file, best_model_snapshot_file)

    return best_val_accuracy, best_epoch, best_iter


def one_stage_train(
    myModel,
    data_reader_trn,
    my_optimizer,
    loss_criterion,
    snapshot_dir,
    log_dir,
    i_iter,
    start_epoch,
    best_val_accuracy=0,
    data_reader_eval=None,
    scheduler=None,
):
    report_interval = cfg.training_parameters.report_interval
    snapshot_interval = cfg.training_parameters.snapshot_interval
    max_iter = cfg.training_parameters.max_iter

    avg_accuracy = 0
    accuracy_decay = 0.99
    best_epoch = 0
    writer = SummaryWriter(log_dir)
    best_iter = i_iter
    iepoch = start_epoch
    snapshot_timer = Timer("m")
    report_timer = Timer("s")

    while i_iter < max_iter:
        iepoch += 1
        for i, batch in enumerate(data_reader_trn):
            i_iter += 1
            if i_iter > max_iter:
                break

            scheduler.step(i_iter)

            my_optimizer.zero_grad()
            add_graph = False
            scores, total_loss, n_sample = compute_a_batch(
                batch,
                myModel,
                eval_mode=False,
                loss_criterion=loss_criterion,
                add_graph=add_graph,
                log_dir=log_dir,
            )
            total_loss.backward()
            accuracy = scores / n_sample
            avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)

            clip_gradients(myModel, i_iter, writer)
            my_optimizer.step()

            if i_iter % report_interval == 0:
                save_a_report(
                    i_iter,
                    total_loss.data[0],
                    accuracy,
                    avg_accuracy,
                    report_timer,
                    writer,
                    data_reader_eval,
                    myModel,
                    loss_criterion,
                )

            if i_iter % snapshot_interval == 0 or i_iter == max_iter:
                best_val_accuracy, best_epoch, best_iter = save_a_snapshot(
                    snapshot_dir,
                    i_iter,
                    iepoch,
                    myModel,
                    my_optimizer,
                    loss_criterion,
                    best_val_accuracy,
                    best_epoch,
                    best_iter,
                    snapshot_timer,
                    data_reader_eval,
                )

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    print(
        "best_acc:%.6f after epoch: %d/%d at iter %d"
        % (best_val_accuracy, best_epoch, iepoch, best_iter)
    )
    sys.stdout.flush()


def evaluate_a_batch(batch, myModel, loss_criterion):
    answer_scores = batch["ans_scores"]
    n_sample = answer_scores.size(0)

    input_answers_variable = Variable(answer_scores.type(torch.FloatTensor))
    if use_cuda:
        input_answers_variable = input_answers_variable.cuda()

    logit_res = one_stage_run_model(batch, myModel)
    predicted_scores = torch.sum(
        compute_score_with_logits(logit_res, input_answers_variable.data)
    )
    total_loss = loss_criterion(logit_res, input_answers_variable)

    return predicted_scores / n_sample, total_loss.data[0]


def compute_a_batch(
    batch, my_model, eval_mode, loss_criterion=None, add_graph=False, log_dir=None
):

    obs_res = batch["ans_scores"]
    obs_res = Variable(obs_res.type(torch.FloatTensor))
    if use_cuda:
        obs_res = obs_res.cuda()

    n_sample = obs_res.size(0)
    logit_res = one_stage_run_model(batch, my_model, eval_mode, add_graph, log_dir)
    predicted_scores = torch.sum(compute_score_with_logits(logit_res, obs_res.data))

    total_loss = None if loss_criterion is None else loss_criterion(logit_res, obs_res)

    return predicted_scores, total_loss, n_sample


def one_stage_eval_model(data_reader_eval, myModel, loss_criterion=None):
    score_tot = 0
    n_sample_tot = 0
    loss_tot = 0
    for idx, batch in enumerate(data_reader_eval):
        score, loss, n_sample = compute_a_batch(
            batch, myModel, eval_mode=True, loss_criterion=loss_criterion
        )
        score_tot += score
        n_sample_tot += n_sample
        if loss is not None:
            loss_tot += loss.data[0] * n_sample
    return score_tot / n_sample_tot, loss_tot / n_sample_tot, n_sample_tot


def one_stage_run_model(batch, my_model, eval_mode, add_graph=False, log_dir=None):
    if eval_mode:
        my_model.eval()
    else:
        my_model.train()

    input_text_seqs = batch["input_seq_batch"]
    input_images = batch["image_feat_batch"]
    input_txt_variable = Variable(input_text_seqs.type(torch.LongTensor))
    image_feat_variable = Variable(input_images)
    if use_cuda:
        input_txt_variable = input_txt_variable.cuda()
        image_feat_variable = image_feat_variable.cuda()

    image_feat_variables = [image_feat_variable]

    image_dim_variable = None
    if "image_dim" in batch:
        image_dims = batch["image_dim"]
        image_dim_variable = Variable(image_dims, requires_grad=False, volatile=False)
        if use_cuda:
            image_dim_variable = image_dim_variable.cuda()

    # check if more than 1 image_feat_batch
    i = 1
    image_feat_key = "image_feat_batch_%s"
    while image_feat_key % str(i) in batch:
        tmp_image_variable = Variable(batch[image_feat_key % str(i)])
        if use_cuda:
            tmp_image_variable = tmp_image_variable.cuda()
        image_feat_variables.append(tmp_image_variable)
        i += 1

    logit_res = my_model(
        input_question_variable=input_txt_variable,
        image_dim_variable=image_dim_variable,
        image_feat_variables=image_feat_variables,
    )

    return logit_res
