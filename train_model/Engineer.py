# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from global_variables.global_variables import use_cuda
from config.config import cfg
from tools.timer import Timer
from torch import autograd


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
    scores = (one_hots * labels)
    return scores


def clip_gradients(model, i_iter, writer, grads_label):

    if grads_label == 'primary':
        max_grad_l2_norm = cfg.training_parameters.max_grad_l2_norm
        clip_norm_mode = cfg.training_parameters.clip_norm_mode
    elif grads_label == 'complement':
        max_grad_l2_norm = cfg.training_parameters.complement_max_grad_l2_norm
        clip_norm_mode = cfg.training_parameters.complement_clip_norm_mode
    else:
        raise NotImplementedError

    if max_grad_l2_norm is not None:
        if clip_norm_mode == 'all':
            norm = nn.utils.clip_grad_norm_(model.parameters(),
                                            max_grad_l2_norm)
            writer.add_scalar('grad_norm_' + grads_label, norm, i_iter)
        elif clip_norm_mode == 'question':
            norm = nn.utils.clip_grad_norm_(
                model.module.question_embedding_models.parameters(),
                max_grad_l2_norm)
            writer.add_scalar('question_grad_norm_' + grads_label, norm, i_iter)
        else:
            raise NotImplementedError


def save_a_report(i_iter,
                  train_losses,
                  train_acc,
                  train_avg_acc,
                  report_timer,
                  writer,
                  data_reader_eval,
                  model,
                  loss_criterions):
    val_batch = next(iter(data_reader_eval))
    val_score, val_losses, n_val_sample = compute_a_batch(
        val_batch, model,
        eval_mode=True,
        loss_criterions=loss_criterions)
    val_acc = val_score / n_val_sample

    val_loss = val_losses[0]
    train_loss = train_losses[0]
    train_comp_loss = None
    val_comp_loss = None

    if len(loss_criterions) == 2:
        val_comp_loss = val_losses[1]
        train_comp_loss = train_losses[1]

    print("iter:", i_iter, "train_loss: %.4f" % train_loss.item(),
          "train_comp_loss: %.4f" % train_comp_loss.item()
          if train_comp_loss is not None else "",
          " train_score: %.4f" % train_acc,
          " avg_train_score: %.4f" % train_avg_acc,
          "val_score: %.4f" % val_acc,
          "val_loss: %.4f" % val_loss.item(),
          "val_comp_loss: %.4f" % val_comp_loss.item()
          if val_comp_loss is not None else "",
          "time(s): % s" % report_timer.end())

    sys.stdout.flush()
    report_timer.start()

    writer.add_scalar('train_loss', train_loss, i_iter)
    writer.add_scalar('train_score', train_acc, i_iter)
    writer.add_scalar('train_score_avg', train_avg_acc, i_iter)
    writer.add_scalar('val_score', val_acc, i_iter)
    writer.add_scalar('val_loss', val_loss.item(), i_iter)

    if train_comp_loss is not None:
        writer.add_scalar('train_comp_loss', train_comp_loss, i_iter)
    if val_comp_loss is not None:
        writer.add_scalar('val_comp_loss', train_loss, i_iter)

    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), i_iter)


def save_a_snapshot(snapshot_dir,
                    i_iter,
                    iepoch,
                    model,
                    optimizer_list,
                    loss_criterions,
                    best_val_accuracy,
                    best_epoch,
                    best_iter,
                    snapshot_timer,
                    data_reader_eval):
    model_snapshot_file = os.path.join(snapshot_dir, "model_%08d.pth" % i_iter)
    model_result_file = os.path.join(snapshot_dir, "result_on_val.txt")
    save_dic = {
        'epoch': iepoch,
        'iter': i_iter,
        'state_dict': model.state_dict(),
        'optimizer': optimizer_list[0].state_dict()
    }

    if len(optimizer_list) == 2:
        save_dic['complement_optimizer'] = optimizer_list[1].state_dict()

    if data_reader_eval is not None:
        val_accuracy, avg_losses, val_sample_tot = one_stage_eval_model(
            data_reader_eval, model,
            loss_criterions=loss_criterions)

        print("i_epoch:", iepoch, "i_iter:", i_iter,
              "val_loss:%.4f" % avg_losses[0],
              "val_comp_loss:%.4f" % avg_losses[1]
              if len(loss_criterions) == 2 else "",
              "val_acc:%.4f" % val_accuracy,
              "runtime: %s" % snapshot_timer.end())
        snapshot_timer.start()
        sys.stdout.flush()

        with open(model_result_file, 'a') as fid:
            fid.write('%d %d %.5f\n' % (iepoch, i_iter, val_accuracy))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = iepoch
            best_iter = i_iter
            best_model_snapshot_file = os.path.join(snapshot_dir,
                                                    "best_model.pth")

        save_dic['best_val_accuracy'] = best_val_accuracy
        torch.save(save_dic, model_snapshot_file)

        if best_iter == i_iter:
            if os.path.exists(best_model_snapshot_file):
                os.remove(best_model_snapshot_file)
            os.link(model_snapshot_file, best_model_snapshot_file)

    return best_val_accuracy, best_epoch, best_iter


def one_stage_train(model,
                    data_reader_trn,
                    optimizer_list,
                    loss_criterions,
                    snapshot_dir, log_dir,
                    i_iter,
                    start_epoch,
                    best_val_accuracy=0,
                    data_reader_eval=None,
                    scheduler_list=None):
    """
    Parameters
    ----------

    loss_criterions: list
    Loss criteria for primary and complement objectives passed in list

    optimizer_list: list
    List of optimizers corresponding to the losses

    scheduler_list: list
    List of schedulers corresponding to the optimizers

    """
    report_interval = cfg.training_parameters.report_interval
    snapshot_interval = cfg.training_parameters.snapshot_interval
    max_iter = cfg.training_parameters.max_iter
    use_complement_loss = cfg.use_complement_loss
    avg_accuracy = 0
    accuracy_decay = 0.99
    best_epoch = 0
    writer = SummaryWriter(log_dir)
    best_iter = i_iter
    iepoch = start_epoch
    snapshot_timer = Timer('m')
    report_timer = Timer('s')

    while i_iter < max_iter:
        iepoch += 1
        for i, batch in enumerate(data_reader_trn):
            i_iter += 1
            if i_iter > max_iter:
                break
            losses = []
            scheduler_list[0].step(i_iter)

            optimizer_list[0].zero_grad()
            add_graph = False

            scores, total_loss, n_sample = compute_a_batch(batch,
                                                           model,
                                                           eval_mode=False,
                                                           loss_criterions=
                                                           loss_criterions[0],
                                                           add_graph=add_graph,
                                                           log_dir=log_dir,
                                                           iter=i_iter)
            total_loss.backward()
            clip_gradients(model, i_iter, writer, 'primary')
            optimizer_list[0].step()
            losses.append(total_loss)

            # ------------------------------------------------------------------
            # Perform the update with complement loss
            # ------------------------------------------------------------------
            if use_complement_loss:
                scheduler_list[1].step(i_iter)
                optimizer_list[1].zero_grad()
                # --------------------------------------------------------------
                # Check gradient Anomaly (i.e. Nan gradients)
                # --------------------------------------------------------------
                with autograd.detect_anomaly():
                    _, comp_loss, _ = compute_a_batch(batch,
                                                      model,
                                                      eval_mode=False,
                                                      loss_criterions=
                                                      loss_criterions[1],
                                                      add_graph=add_graph,
                                                      log_dir=log_dir,
                                                      iter=i_iter)
                    comp_loss.backward()
                clip_gradients(model, i_iter, writer, 'complement')
                optimizer_list[1].step()
                losses.append(comp_loss)

            accuracy = scores / n_sample
            avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)

            if i_iter % report_interval == 0:
                save_a_report(i_iter, losses, accuracy,
                              avg_accuracy, report_timer, writer,
                              data_reader_eval,
                              model, loss_criterions)

            if i_iter % snapshot_interval == 0 or i_iter == max_iter:
                best_val_accuracy, best_epoch, best_iter = save_a_snapshot(
                    snapshot_dir,
                    i_iter,
                    iepoch,
                    model,
                    optimizer_list,
                    loss_criterions,
                    best_val_accuracy,
                    best_epoch,
                    best_iter,
                    snapshot_timer,
                    data_reader_eval)

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    print("best_acc:%.6f after epoch: %d/%d at iter %d" % \
          (best_val_accuracy, best_epoch, iepoch, best_iter))
    sys.stdout.flush()


def evaluate_a_batch(batch, model, loss_criterions):
    """
    Parameters
    ----------
    loss_criterions: Could be either a list or single loss (nn.Module) object.
    """
    answer_scores = batch['ans_scores']
    n_sample = answer_scores.size(0)

    input_answers_variable = Variable(answer_scores.type(torch.FloatTensor))
    if use_cuda:
        input_answers_variable = input_answers_variable.cuda()

    logit_res = one_stage_run_model(batch, model)
    predicted_scores = torch.sum(compute_score_with_logits(
        logit_res,
        input_answers_variable.data))

    losses = None
    if loss_criterions is not None:
        if isinstance(loss_criterions, list):
            losses = []
            for loss in loss_criterions:
                losses.append(loss(logit_res, input_answers_variable))
        elif isinstance(loss_criterions, nn.Module):
            losses = loss_criterions(logit_res, input_answers_variable)
    return predicted_scores / n_sample, losses


def compute_a_batch(batch, my_model, eval_mode, loss_criterions=None,
                    add_graph=False, log_dir=None, iter=None):
    """
    Parameters
    ----------
    loss_criterions: Could be either a list or single loss (nn.Module) object.
    """
    obs_res = batch['ans_scores']
    obs_res = Variable(obs_res.type(torch.FloatTensor))
    if use_cuda:
        obs_res = obs_res.cuda()

    n_sample = obs_res.size(0)
    logit_res = one_stage_run_model(batch, my_model, eval_mode,
                                    add_graph, log_dir)
    predicted_scores = torch.sum(compute_score_with_logits(
        logit_res, obs_res.data))

    losses = None
    if loss_criterions is not None:
        if isinstance(loss_criterions, list):
            losses = []
            for loss in loss_criterions:
                losses.append(loss(logit_res, obs_res, iter))
        elif isinstance(loss_criterions, nn.Module):
            losses = loss_criterions(logit_res, obs_res, iter)

    return predicted_scores, losses, n_sample


def one_stage_eval_model(data_reader_eval, model, loss_criterions=None):
    """
    Parameters
    ----------
    loss_criterions: Could be either a list or single loss (nn.Module) object.
    """
    score_tot = 0
    n_sample_tot = 0
    losses = [0, 0]
    for idx, batch in enumerate(data_reader_eval):
        score, temp_losses, n_sample = compute_a_batch(batch, model,
                                                       eval_mode=True,
                                                       loss_criterions=
                                                       loss_criterions)
        score_tot += score
        n_sample_tot += n_sample
        if temp_losses is not None:
            if isinstance(temp_losses, list):
                losses[0] += temp_losses[0].item() * n_sample
                if len(temp_losses) == 2:
                    losses[1] = temp_losses[1].item() * n_sample
            else:
                losses[0] += temp_losses.item() * n_sample

    if isinstance(loss_criterions, nn.Module):
        losses = losses[0] / n_sample_tot  # send a single value not list
    elif isinstance(loss_criterions, list):
        losses = [loss / n_sample_tot for loss in losses]
    else:
        losses = None  # loss_criterions is none

    return score_tot / n_sample_tot, losses, n_sample_tot


def one_stage_run_model(batch, my_model, eval_mode,
                        add_graph=False, log_dir=None):
    if eval_mode:
        my_model.eval()
    else:
        my_model.train()

    input_text_seqs = batch['input_seq_batch']
    input_images = batch['image_feat_batch']
    input_txt_variable = Variable(input_text_seqs.type(torch.LongTensor))
    image_feat_variable = Variable(input_images)
    if use_cuda:
        input_txt_variable = input_txt_variable.cuda()
        image_feat_variable = image_feat_variable.cuda()

    image_feat_variables = [image_feat_variable]

    image_dim_variable = None
    if 'image_dim' in batch:
        image_dims = batch['image_dim']
        image_dim_variable = Variable(image_dims,
                                      requires_grad=False,
                                      volatile=False)
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

    logit_res = my_model(input_question_variable=input_txt_variable,
                         image_dim_variable=image_dim_variable,
                         image_feat_variables=image_feat_variables)

    return logit_res
