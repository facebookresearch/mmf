import os
import gc
import sys
import torch
import copy
import shutil
import logging
import json
import torch.nn as nn
import numpy as np
import timeit
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate
from global_variables.global_variables import use_cuda
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from config.config import cfg
from tqdm import tqdm


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

def obtain_vocabs(cfg):
    q_vocab_path = os.path.join(cfg.data.data_root_dir, cfg.data.vocab_question_file)
    a_vocab_path = os.path.join(cfg.data.data_root_dir, cfg.data.vocab_answer_file)

    q_vocab = [l.rstrip() for l in tuple(open(q_vocab_path))]
    q_vocab.extend(["<start>", "<end>"])
    a_vocab = [l.rstrip() for l in tuple(open(a_vocab_path))]
    return q_vocab, a_vocab

def one_stage_train(
    myModel,
    data_reader_trn,
    myOptimizer,
    loss_criterion,
    snapshot_dir,
    log_dir,
    i_iter,
    start_epoch,
    data_reader_eval=None,
    scheduler=None,
):

    clip_norm_mode = cfg.training_parameters.clip_norm_mode
    max_grad_l2_norm = cfg.training_parameters.max_grad_l2_norm
    report_interval = cfg.training_parameters.report_interval
    snapshot_interval = cfg.training_parameters.snapshot_interval
    max_iter = cfg.training_parameters.max_iter

    is_failure_prediction = hasattr(myModel, "failure_predictor")
    is_question_consistency = hasattr(myModel, "question_consistency")

    vocab, ans_vocab = obtain_vocabs(cfg)

    if isinstance(myModel, torch.nn.DataParallel):
        is_failure_prediction = hasattr(myModel.module, "failure_predictor")
        is_question_consistency = hasattr(myModel.module, "question_consistency")

    # Set model to train
    myModel = myModel.train()

    avg_accuracy = 0
    accuracy_decay = 0.99
    best_val_accuracy = 0
    best_val_precision = 0.0
    writer = SummaryWriter(log_dir)
    best_iter = i_iter
    iepoch = start_epoch
    start = timeit.default_timer()
    confusion_mat = np.zeros((2, 2))
    val_confusion_mat = np.zeros((2, 2))

    while i_iter < max_iter:
        n_sample_tot = 0

        start_iter = timeit.default_timer()
        iepoch += 1
        for i, batch in enumerate(data_reader_trn):
            i_iter += 1
            if i_iter > max_iter:
                break

            scheduler.step(i_iter)

            answer_scores = batch["ans_scores"]
            answer_scores_cuda = batch["ans_scores"].cuda()
            n_sample = answer_scores.size(0)
            n_sample_tot += n_sample
            myOptimizer.zero_grad()

            add_graph = False

            myModel = myModel.train()
            if is_failure_prediction and not is_question_consistency:
                _return_dict = one_stage_run_model(batch, myModel, add_graph, log_dir)
                logit_res = _return_dict["logits"]
                fp_return_dict = _return_dict["fp_return_dict"]
                confidence_fp = fp_return_dict["confidence"]

            elif is_question_consistency and not is_failure_prediction:
                _return_dict = one_stage_run_model(batch, myModel, add_graph, log_dir)
                logit_res = _return_dict["logits"]
                qc_return_dict = _return_dict["qc_return_dict"]
                qc_loss = torch.mean(qc_return_dict["qc_loss"], 0)

            elif is_failure_prediction and is_question_consistency:
                _return_dict = one_stage_run_model(batch, myModel, add_graph, log_dir)
                logit_res = _return_dict["logits"]
                fp_return_dict = _return_dict["fp_return_dict"]
                confidence_fp = fp_return_dict["confidence"]
                qc_return_dict = _return_dict["qc_return_dict"]
                qc_loss = qc_return_dict["qc_loss"]

            else:
                logit_res = one_stage_run_model(batch, myModel, add_graph, log_dir)[
                    "logits"
                ]

            input_answers_variable = Variable(answer_scores.type(torch.FloatTensor))
            if use_cuda:
                input_answers_variable = input_answers_variable.cuda()

            total_loss = loss_criterion(logit_res, input_answers_variable)

            if is_failure_prediction and not is_question_consistency:
                normalized_logits = masked_unk_softmax(logit_res, 1, 0)
                answers_fp = (
                    torch.max(normalized_logits, 1)[1]
                    == torch.max(answer_scores_cuda, 1)[1]
                )
                fp_loss = F.cross_entropy(
                    input=confidence_fp,
                    target=answers_fp.long(),
                    weight=torch.Tensor([1.0, 1.0]).cuda(),
                )
                total_loss += cfg.training_parameters.fp_lambda * fp_loss
                qc_loss = 0

            elif is_question_consistency and not is_failure_prediction:
                total_loss += cfg.training_parameters.qc_lambda * qc_loss
                fp_loss = 0

            elif is_question_consistency and is_failure_prediction:
                normalized_logits = masked_unk_softmax(logit_res, 1, 0)
                answers_fp = (
                    torch.max(normalized_logits, 1)[1]
                    == torch.max(answer_scores_cuda, 1)[1]
                )
                fp_loss = F.cross_entropy(
                    input=confidence_fp,
                    target=answers_fp.long(),
                    weight=torch.Tensor([1.0, 1.0]).cuda(),
                )
                total_loss += cfg.training_parameters.fp_lambda * fp_loss
                total_loss += cfg.training_parameters.qc_lambda * qc_loss

            else:
                qc_loss = 0
                fp_loss = 0

            if is_failure_prediction:
                # Thresholding at zero is just computing CM
                batch_cm = thresholding(
                    confidence_fp, answers_fp, ttype="failure_prediction", threshold=0.0
                )
                confusion_mat += batch_cm

            print("{}_{}: QC: {:.4f} FP: {:.4f} T: {:.4f}".format(
                iepoch, i_iter, float(qc_loss), float(fp_loss), float(total_loss)
            ))

            total_loss.backward()
            if max_grad_l2_norm is not None:
                if clip_norm_mode == "all":
                    norm = nn.utils.clip_grad_norm_(
                        myModel.parameters(), max_grad_l2_norm
                    )
                    writer.add_scalar("grad_norm", norm, i_iter)
                elif clip_norm_mode == "question":
                    norm = nn.utils.clip_grad_norm_(
                        myModel.module.question_embedding_models.parameters(),
                        max_grad_l2_norm,
                    )
                    writer.add_scalar("question_grad_norm", norm, i_iter)
                elif clip_norm_mode == "none":
                    pass
                else:
                    raise NotImplementedError
            myOptimizer.step()

            if (
                cfg.model.question_consistency.cycle
                and i_iter > cfg["model"]["question_consistency"]["activation_iter"]
            ):
                cycle_batch = {}
                for _k, _v in batch.items():
                    cycle_batch[_k] = _v.clone()
                generated_questions = qc_return_dict["sampled_ids"].clone()
                # Preprocess to remove start and end
                generated_questions[generated_questions == len(vocab)-2] = 0
                generated_questions[generated_questions == len(vocab)-1] = 0

                # First letter cannot be unk
                generated_questions = torch.cat(
                    [
                        generated_questions.narrow(
                            1, 1, generated_questions.shape[1] - 1
                        ),
                        generated_questions.narrow(1, 0, 1),
                    ],
                    1,
                )

                # Gating Mechanism
                if cfg["model"]["question_consistency"]["gating_th"] > 0:
                    detached_g_q = generated_questions.clone().detach()
                    detached_g_emb = myModel.question_consistency.embed(
                        detached_g_q
                    ).sum(1)

                    detached_o_q = batch["input_seq_batch"].long().cuda()
                    detached_o_emb = myModel.question_consistency.embed(
                        detached_o_q
                    ).sum(1)

                    cosine_similarity = F.cosine_similarity(
                        detached_g_emb, detached_o_emb
                    )
                    allowed_indices = (
                        cosine_similarity
                        > cfg["model"]["question_consistency"]["gating_th"]
                    )
                    print(
                        "Allowed Batches {}".format(allowed_indices.sum().cpu().item())
                    )
                else:
                    allowed_indices = torch.ones(len(generated_questions)).long()

                cycle_batch["input_seq_batch"] = generated_questions
                cycle_return_dict = one_stage_run_model(cycle_batch, myModel)

                if cfg["model"]["question_consistency"]["vqa_gating"]:
                    allowed_indices = (
                        cycle_return_dict["logits"].max(1)[1]
                        == answer_scores_cuda.max(1)[1]
                    )

                if allowed_indices.sum() > -1:
                    cycle_vqa_loss = cfg.training_parameters.cc_lambda * loss_criterion(
                        cycle_return_dict["logits"][allowed_indices],
                        cycle_batch["ans_scores"][allowed_indices].cuda(),
                    )

                    # perform backward pass
                    cycle_vqa_loss.backward()
                    print(
                        "CL: {:.4f} Pass: [{}/512]".format(
                            cycle_vqa_loss, allowed_indices.sum().cpu().item()
                        )
                    )
                    myOptimizer.step()

            scores = torch.sum(
                compute_score_with_logits(logit_res, input_answers_variable.data)
            )
            accuracy = scores / n_sample
            avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)

            if i_iter % report_interval == 0:
                cur_loss = total_loss.item()
                end_iter = timeit.default_timer()
                time = end_iter - start_iter
                start_iter = timeit.default_timer()
                val_batch = next(iter(data_reader_eval))
                val_score, val_loss = evaluate_a_batch(
                    val_batch, myModel, loss_criterion
                )

                print(
                    "iter:",
                    i_iter,
                    "train_loss: %.4f" % cur_loss,
                    " train_score: %.4f" % accuracy,
                    " qc_loss: %.4f" % qc_loss,
                    " fp_loss: %.4f" % fp_loss,
                    " avg_train_score: %.4f" % avg_accuracy,
                    "val_score: %.4f" % val_score,
                    "val_loss: %.4f" % val_loss,
                    "time(s): %.1f" % time,
                )
                sys.stdout.flush()

                cm_stat_dict = print_classification_report(
                    confusion_mat, threshold=0.0, return_dict=True
                )

                for k, v in cm_stat_dict.items():
                    writer.add_scalar("train_" + k, v, i_iter)

                writer.add_scalar("train_loss", cur_loss, i_iter)
                writer.add_scalar("train_score", accuracy, i_iter)
                writer.add_scalar("train_score_avg", avg_accuracy, i_iter)
                writer.add_scalar("val_score", val_score, i_iter)
                writer.add_scalar("val_loss", val_loss, i_iter)

            if (i_iter % snapshot_interval == 1 or i_iter == max_iter) and i_iter != 1:
                ##evaluate the model when finishing one epoch
                if data_reader_eval is not None:
                    val_accuracy, upbound_acc, val_sample_tot, val_confusion_mat = one_stage_eval_model(
                        data_reader_eval,
                        myModel,
                        return_cm=True,
                        i_iter=i_iter,
                        log_dir=log_dir,
                    )
                    val_precision = print_classification_report(
                        val_confusion_mat, threshold=0.0, return_dict=True
                    )["prec"]

                    end = timeit.default_timer()
                    epoch_time = end - start
                    start = timeit.default_timer()
                    print(
                        "i_epoch:",
                        iepoch,
                        "i_iter:",
                        i_iter,
                        "val_acc:%.4f" % val_accuracy,
                        "runtime(s):%d" % epoch_time,
                    )
                    sys.stdout.flush()

                model_snapshot_file = os.path.join(
                    snapshot_dir, "model_%08d.pth" % i_iter
                )
                model_result_file = os.path.join(
                    snapshot_dir, "result_%08d.txt" % i_iter
                )
                torch.save(
                    {
                        "epoch": iepoch,
                        "iter": i_iter,
                        "state_dict": myModel.state_dict(),
                        "optimizer": myOptimizer.state_dict(),
                    },
                    model_snapshot_file,
                )
                with open(model_result_file, "w") as fid:
                    fid.write("%d:%.5f\n" % (iepoch, val_accuracy * 100))

                if is_failure_prediction:
                    if val_precision > best_val_precision:
                        best_val_precision = val_precision
                        best_epoch = iepoch
                        best_iter = i_iter
                        best_model_snapshot_file = os.path.join(
                            snapshot_dir, "best_model.pth"
                        )
                        shutil.copy(model_snapshot_file, best_model_snapshot_file)

                else:
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        best_epoch = iepoch
                        best_iter = i_iter
                        best_model_snapshot_file = os.path.join(
                            snapshot_dir, "best_model.pth"
                        )
                        shutil.copy(model_snapshot_file, best_model_snapshot_file)

        print("training" + "=" * 45)
        print_classification_report(confusion_mat, 0.0)
        confusion_mat = np.zeros((2, 2))
        print("=" * 53)
        gc.collect()

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    print(
        "best_acc:%.6f after epoch: %d/%d at iter %d"
        % (best_val_accuracy, best_epoch, iepoch, best_iter)
    )
    sys.stdout.flush()


def evaluate_a_batch(batch, myModel, loss_criterion):
    answer_scores = batch["ans_scores"]
    answer_scores_cuda = answer_scores.cuda()
    n_sample = answer_scores.size(0)

    input_answers_variable = Variable(answer_scores.type(torch.FloatTensor))
    input_answers_variable = (
        input_answers_variable.cuda() if use_cuda else input_answers_variable
    )

    # Set model to eval
    myModel = myModel.eval()

    is_failure_prediction = hasattr(myModel, "failure_prediction")
    is_question_consistency = hasattr(myModel, "question_consistency")

    if isinstance(myModel, torch.nn.DataParallel):
        is_failure_prediction = (
            True if hasattr(myModel.module, "failure_predictor") else False
        )
        is_question_consistency = (
            True if hasattr(myModel.module, "question_consistency") else False
        )

    if is_failure_prediction and not is_question_consistency:
        _return_dict = one_stage_run_model(batch, myModel)
        logit_res = _return_dict["logits"]
        fp_return_dict = _return_dict["fp_return_dict"]
        confidence_fp = fp_return_dict["confidence"]

    elif is_question_consistency and not is_failure_prediction:
        _return_dict = one_stage_run_model(batch, myModel)
        logit_res = _return_dict["logits"]
        qc_return_dict = _return_dict["qc_return_dict"]
        qc_loss = torch.mean(qc_return_dict["qc_loss"])

    elif is_question_consistency and is_failure_prediction:
        _return_dict = one_stage_run_model(batch, myModel)
        logit_res = _return_dict["logits"]
        fp_return_dict = _return_dict["fp_return_dict"]
        confidence_fp = fp_return_dict["confidence"]
        qc_return_dict = _return_dict["qc_return_dict"]
        qc_loss = qc_return_dict["qc_loss"]

    else:
        logit_res = one_stage_run_model(batch, myModel)["logits"]

    predicted_scores = torch.sum(
        compute_score_with_logits(logit_res, input_answers_variable.data)
    )

    total_loss = loss_criterion(logit_res, input_answers_variable)

    if is_failure_prediction and not is_question_consistency:
        total_loss = 0
        normalized_logits = masked_unk_softmax(logit_res, 1, 0)
        answers_fp = (
            torch.max(normalized_logits, 1)[1] == torch.max(answer_scores_cuda, 1)[1]
        )
        total_loss += F.cross_entropy(input=confidence_fp, target=answers_fp.long())

    if is_question_consistency and not is_failure_prediction:
        total_loss = 0
        total_loss += qc_loss

    elif is_failure_prediction and is_question_consistency:
        normalized_logits = masked_unk_softmax(logit_res, 1, 0)
        answers_fp = (
            torch.max(normalized_logits, 1)[1] == torch.max(answer_scores_cuda, 1)[1]
        )
        total_loss += F.cross_entropy(input=confidence_fp, target=answers_fp.long())
        total_loss += qc_loss

    gc.collect()
    return predicted_scores / n_sample, total_loss.item()


def one_stage_eval_model(
    data_reader_eval,
    myModel,
    thresholding_type=None,
    threshold=0.5,
    return_cm=False,
    i_iter=None,
    log_dir=None,
):

    val_score_tot = 0
    val_sample_tot = 0
    upbound_tot = 0
    writer = SummaryWriter(log_dir)

    # Set model to eval
    myModel = myModel.eval()
    vocab, ans_vocab = obtain_vocabs(cfg)

    # Make dict to store generated questions
    gq_dict = {"annotations": [], "answers": []}

    def store_questions(sampled_ids, batch):
        sampled_ids = sampled_ids.data.cpu().numpy()
        questions = [
            [vocab[idx] for idx in sampled_ids[j]] for j in range(len(sampled_ids))
        ]
        images = batch["image_id"]
        orig_answers = batch["answer_label_batch"].data.cpu().numpy()
        for jdx, (q, img, oa) in enumerate(zip(questions, images, orig_answers)):
            gq_dict["annotations"] += [{"image_id": int(img), "caption": " ".join(q)}]
            gq_dict["answers"] += [{"image_id": int(img), "caption": ans_vocab[oa]}]

    confusion_mat = (
        np.zeros((len(threshold), 2, 2))
        if type(threshold) == list
        else np.zeros((2, 2))
    )
    is_failure_prediction = True if hasattr(myModel, "failure_predictor") else False
    is_question_consistency = (
        True if hasattr(myModel, "question_consistency") else False
    )

    if isinstance(myModel, torch.nn.DataParallel):
        is_failure_prediction = (
            True if hasattr(myModel.module, "failure_predictor") else False
        )
        is_question_consistency = (
            True if hasattr(myModel.module, "question_consistency") else False
        )

    for idx, batch in tqdm(enumerate(data_reader_eval)):
        answer_scores = batch["ans_scores"]
        n_sample = answer_scores.size(0)
        answer_scores = answer_scores.cuda() if use_cuda else answer_scores

        if is_failure_prediction and not is_question_consistency:
            _return_dict = one_stage_run_model(batch, myModel)
            logit_res = _return_dict["logits"]
            fp_return_dict = _return_dict["fp_return_dict"]
            confidence_fp = fp_return_dict["confidence"]

        elif is_question_consistency and not is_failure_prediction:
            _return_dict = one_stage_run_model(batch, myModel)
            logit_res = _return_dict["logits"]
            qc_return_dict = _return_dict["qc_return_dict"]
            if "sampled_ids" in qc_return_dict.keys():
                sampled_ids = qc_return_dict["sampled_ids"]
                store_questions(sampled_ids, batch)

        elif is_question_consistency and is_failure_prediction:
            _return_dict = one_stage_run_model(batch, myModel)
            logit_res = _return_dict["logits"]
            fp_return_dict = _return_dict["fp_return_dict"]
            confidence_fp = fp_return_dict["confidence"]
            qc_return_dict = _return_dict["qc_return_dict"]
            if "sampled_ids" in qc_return_dict.keys():
                sampled_ids = qc_return_dict["sampled_ids"]
                store_questions(sampled_ids, batch)
        else:
            logit_res = one_stage_run_model(batch, myModel)["logits"]

        predicted_scores = torch.sum(
            compute_score_with_logits(logit_res, answer_scores)
        )
        upbound = torch.sum(torch.max(answer_scores, dim=1)[0])

        if thresholding_type is not None and thresholding_type != "failure_prediction":

            if type(threshold) == list and len(threshold) > 0:
                for _th_idx, _th in enumerate(threshold):
                    normalized_logits = masked_unk_softmax(logit_res, 1, 0)
                    batch_cm = thresholding(
                        normalized_logits,
                        answer_scores,
                        thresholding_type,
                        threshold[_th_idx],
                    )
                    confusion_mat[_th_idx] += batch_cm

            else:
                normalized_logits = masked_unk_softmax(logit_res, 1, 0)
                batch_cm = thresholding(
                    normalized_logits, answer_scores, thresholding_type, threshold
                )
                confusion_mat += batch_cm

        if is_failure_prediction:
            normalized_logits = masked_unk_softmax(logit_res, 1, 0)
            answers_fp = (
                torch.max(normalized_logits, 1)[1]
                == torch.max(answer_scores.cuda(), 1)[1]
            )
            fp_pred = confidence_fp

            # Thresholding at zero is just computing CM
            batch_cm = thresholding(
                fp_pred, answers_fp, ttype="failure_prediction", threshold=0.0
            )
            confusion_mat += batch_cm

        cm_stat_dict = print_classification_report(
            confusion_mat, threshold=0.0, return_dict=True
        )

        for k, v in cm_stat_dict.items():
            if not v != v:
                writer.add_scalar("val_" + k, v, i_iter)

        val_score_tot += predicted_scores
        val_sample_tot += n_sample
        upbound_tot += upbound

    gc.collect()

    if is_question_consistency:
        np.save(os.path.join(log_dir, "gq_{}.npy".format(i_iter)), np.array(gq_dict))

    if is_failure_prediction:
        print("validation" + "=" * 45)
        print_classification_report(confusion_mat, threshold=0.0)
        print("=" * 55)

    else:
        if not type(threshold) == list:
            print_classification_report(confusion_mat, threshold)

        else:
            for _th_idx, _th in enumerate(threshold):
                print_classification_report(confusion_mat[_th_idx], _th)

    if return_cm:
        return (
            val_score_tot / val_sample_tot,
            upbound_tot / val_sample_tot,
            val_sample_tot,
            confusion_mat,
        )

    return val_score_tot / val_sample_tot, upbound_tot / val_sample_tot, val_sample_tot

def print_classification_report(confusion_matrix, threshold, return_dict=False):
    tn, fp, fn, tp = confusion_matrix.ravel()
    prec = tp / float(tp + fp)
    rec = tp / float(tp + fn)
    f1 = 2 * prec * rec / float(prec + rec)
    acc = (tp + tn) / float(tp + tn + fp + fn)
    nacc = 0.5 * tp / float(tp + fn) + 0.5 * tn / float(tn + fp)
    fmt_string = "{:.4f}," * 11
    fmt_string = fmt_string[:-1]
    n_samples = tp + fp + fn + tn

    label_string = "threshold, f1, prec, rec, acc, nacc, tp, tn, fp, fn, n_samples"
    value_list = [threshold, f1, prec, rec, acc, nacc, tp, tn, fp, fn, n_samples]

    if return_dict:
        return dict(
            zip([key.replace(" ", "") for key in label_string.split(",")], value_list)
        )

    else:
        print("threshold, f1, prec, rec, acc, nacc, tp, tn, fp, fn, n_samples")
        print(fmt_string.format(*value_list))


def thresholding(pred, answers, ttype="vanilla", threshold=0.5, return_indices=False):
    """
    pred = 2d array with prob distribution B x 3129
    answers = 2d one-hot encoded array B x 3129
    """

    def get_indices(th_confidence, correct_numpy, confidence, ans_indices, indices):
        return_dict = {
            "fp_idx": np.argwhere(
                np.logical_and(th_confidence == 1, correct_numpy == 0) == True
            ).flatten(),
            "fn_idx": np.argwhere(
                np.logical_and(th_confidence == 0, correct_numpy == 1) == True
            ).flatten(),
            "tp_idx": np.argwhere(
                np.logical_and(th_confidence == 1, correct_numpy == 1) == True
            ).flatten(),
            "tn_idx": np.argwhere(
                np.logical_and(th_confidence == 0, correct_numpy == 0) == True
            ).flatten(),
            "confidence": confidence.cpu()
            .data.numpy()
            .flatten()
            .astype(np.float64)
            .tolist(),
            "ans_indices": ans_indices.cpu()
            .data.numpy()
            .flatten()
            .astype(np.int32)
            .tolist(),
            "pred_indices": indices.cpu()
            .data.numpy()
            .flatten()
            .astype(np.int32)
            .tolist(),
        }
        return return_dict

    if ttype == "entropy":
        pred_np = pred.cpu().data.numpy()
        ent = np.array([entropy(pred_np[i]) for i in range(pred_np.shape[0])])
        confidence, indices = torch.max(pred, 1)
        ans_indices = torch.max(answers, 1)[1]
        th_confidence = ent < threshold
        th_confidence = th_confidence.cpu().data.numpy()
        correct = ans_indices == indices
        correct_numpy = correct.cpu().data.numpy()

        if return_indices == True:
            return_dict = get_indices(
                th_confidence, correct_numpy, confidence, ans_indices, indices
            )
            return confusion_matrix(correct_numpy, th_confidence), return_dict

        else:
            return confusion_matrix(correct.cpu().data.numpy(), th_confidence)

    elif ttype == "vanilla":
        confidence, indices = torch.max(pred, 1)
        ans_indices = torch.max(answers, 1)[1]
        th_confidence = confidence > threshold
        th_confidence = th_confidence.cpu().data.numpy()
        correct = ans_indices == indices
        correct_numpy = correct.cpu().data.numpy()

        if return_indices == True:
            return_dict = get_indices(
                th_confidence, correct_numpy, confidence, ans_indices, indices
            )
            return confusion_matrix(correct_numpy, th_confidence), return_dict

        else:
            return confusion_matrix(correct_numpy, th_confidence)

    elif ttype == "failure_prediction":
        confidence, indices = torch.max(pred, 1)
        ans_indices = answers.cpu().data.numpy()
        correct_numpy = indices.cpu().data.numpy()
        # Add dummy ans and pred indices, we'll update them in the eval function later

        if return_indices == True:
            return_dict = get_indices(
                correct_numpy,
                ans_indices,
                confidence,
                ans_indices=torch.ones_like(confidence),
                indices=torch.ones_like(confidence),
            )
            return confusion_matrix(ans_indices, correct_numpy), return_dict

        return confusion_matrix(ans_indices, correct_numpy)

    elif ttype == "uncertainty":
        uncertainties, max_indices = torch.max(pred, 1)
        ans_indices = answers.cpu().data.numpy()
        correct = uncertainties < threshold
        correct_numpy = correct.cpu().data.numpy()
        # Add dummy ans and pred indices, we'll update them in the eval function later

        if return_indices == True:
            return_dict = get_indices(
                correct_numpy,
                ans_indices,
                uncertainties,
                ans_indices=torch.ones_like(uncertainties),
                indices=torch.ones_like(uncertainties),
            )
            return confusion_matrix(ans_indices, correct_numpy), return_dict

        return confusion_matrix(ans_indices, correct_numpy)

    else:
        raise NotImplementedError(
            "Thresholding type {} not implemented".format(ttype)
        )

def one_stage_run_model(batch, myModel, add_graph=False, log_dir=None,
                        normalize=False):
    input_text_seqs = batch["input_seq_batch"]
    input_images = batch["image_feat_batch"]
    input_txt_variable = Variable(input_text_seqs.type(torch.LongTensor))
    input_txt_variable = input_txt_variable.cuda() if use_cuda else input_txt_variable

    if isinstance(input_images, list):
        input_images = input_images[0]

    image_feat_variable = Variable(input_images)
    image_feat_variable = (
        image_feat_variable.cuda() if use_cuda else image_feat_variable
    )
    image_feat_variables = [image_feat_variable]

    image_dim_variable = None
    if "image_dim" in batch:
        image_dims = batch["image_dim"]
        image_dim_variable = Variable(image_dims, requires_grad=False, volatile=False)
        image_dim_variable = (
            image_dim_variable.cuda() if use_cuda else image_dim_variable
        )

    # check if more than 1 image_feat_batch
    i = 1
    image_feat_key = "image_feat_batch_%s"
    while image_feat_key % str(i) in batch:
        tmp_image_variable = Variable(batch[image_feat_key % str(i)])
        tmp_image_variable = (
            tmp_image_variable.cuda() if use_cuda else tmp_image_variable
        )
        image_feat_variables.append(tmp_image_variable)
        i += 1

    return_dict = myModel(
        input_question_variable=input_txt_variable,
        image_dim_variable=image_dim_variable,
        image_feat_variables=image_feat_variables,
        batch=batch,
    )

    if add_graph:
        with SummaryWriter(log_dir=log_dir, comment="basicblock") as w:
            w.add_graph(
                myModel, (input_txt_variable, image_dim_variable, image_feat_variables)
            )

    if normalize:
        return_dict["logits"] = masked_unk_softmax(return_dict["logits"], 1, 0)

    return return_dict
