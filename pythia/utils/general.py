import yaml
import collections

from bisect import bisect
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
            norm = nn.utils.clip_grad_norm_(model.parameters(),
                                            max_grad_l2_norm)

            writer.add_scalars({'grad_norm': norm}, i_iter)

        elif clip_norm_mode == 'question':
            question_embedding = model.module.question_embedding_module
            norm = nn.utils.clip_grad_norm(question_embedding.parameters(),
                                           max_grad_l2_norm)

            writer.add_scalars({'question_grad_norm': norm}, i_iter)
        else:
            raise NotImplementedError("Clip norm mode %s not implemented"
                                      % clip_norm_mode)


def ckpt_name_from_core_args(config):
    return ("%s_%s_%s_%d" % (config['tasks'], config['datasets'],
                             config['model'], config['seed']))


def foldername_from_config_override(args):
    cfg_override = None
    if hasattr(args, 'config_override'):
        cfg_override = args.config_override
    elif 'config_override' in args:
        cfg_override = args['config_override']

    folder_name = ''
    if cfg_override is not None and len(cfg_override) > 0:
        folder_name = yaml.safe_dump(cfg_override, default_flow_style=True)
        folder_name = folder_name.replace(':', '.').replace('\n', ' ')
        folder_name = folder_name.replace('/', '_')
        folder_name = ' '.join(folder_name.split())
        folder_name = folder_name.replace('. ', '.').replace(' ', '_')
        folder_name = '_' + folder_name
    return folder_name


def get_optimizer_parameters(model, config):
    parameters = model.parameters()

    has_custom = hasattr(model, 'get_optimizer_parameters')
    if has_custom:
        parameters = model.get_optimizer_parameters(config)

    is_parallel = isinstance(model, nn.DataParallel)

    if is_parallel and hasattr(model.module, 'get_optimizer_parameters'):
        parameters = model.module.get_optimizer_parameters(config)

    return parameters


def dict_to_string(dictionary):
    logs = []
    if dictionary is None:
        return ""
    for key, val in dictionary.items():
        if hasattr(val, 'item'):
            val = val.item()
        if key.count('_') == 2:
            key = key[key.find('_') + 1:]
        logs.append('%s: %s' % (key, val))

    return ", ".join(logs)


def nested_dict_update(dictionary, update):
    """Updates a dictionary with other dictionary recursively.

    Parameters
    ----------
    dictionary : dict
        Dictionary to be updated.
    update : dict
        Dictionary which has to be added to original one.

    Returns
    -------
    dict
        Updated dictionary.
    """
    for k, v in update.items():
        if isinstance(v, collections.Mapping):
            dictionary[k] = nested_dict_update(dictionary.get(k, {}), v)
        else:
            dictionary[k] = v
    return dictionary
