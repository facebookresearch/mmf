import bisect

from torch import nn


def lr_lambda_update(i_iter, cfg):
    if i_iter <= cfg['training_parameters']['wu_iters']:
        alpha = float(i_iter) / float(cfg['training_parameters']['wu_iters'])
        return cfg['training_parameters']['wu_factor'] * (1. - alpha) + alpha
    else:
        idx = bisect(cfg['training_parameters']['lr_steps'], i_iter)
        return pow(cfg['training_parameters']['lr_ratio'], idx)


def clip_gradients(model, i_iter, writer, config):
    # TODO: Fix question model retrieval
    max_grad_l2_norm = config['training_parameters']['max_grad_l2_norm']
    clip_norm_mode = config['training_parameters']['clip_norm_mode']

    if max_grad_l2_norm is not None:
        if clip_norm_mode == 'all':
            norm = nn.utils.clip_grad_norm(model.parameters(),
                                           max_grad_l2_norm)

            writer.add_scalar('grad_norm', norm, i_iter)

        elif clip_norm_mode == 'question':
            question_embedding = model.module.question_embedding_module
            norm = nn.utils.clip_grad_norm(question_embedding.parameters(),
                                           max_grad_l2_norm)

            writer.add_scalar('question_grad_norm', norm, i_iter)
        else:
            raise NotImplementedError("Clip norm mode %s not implemented"
                                      % clip_norm_mode)


def ckpt_name_from_core_args(config):
    return ("%s_%s_%d" % (config['task'], config['model'], config['seed']))
