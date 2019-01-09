import os
import torch
import yaml

from pythia.utils.general import ckpt_name_from_core_args, \
                                 foldername_from_config_override
from pythia.core.registry import registry


class Checkpoint:
    def __init__(self, trainer):
        '''
        Generates a path for saving model which can also be used for resuming
        from a checkpoint.
        '''
        self.trainer = trainer

        self.config = self.trainer.config
        self.save_loc = self.config['save_loc']

        self.ckpt_foldername = ckpt_name_from_core_args(self.config)
        self.ckpt_foldername += foldername_from_config_override(
            self.trainer.args
        )

        self.ckpt_prefix = ""
        if hasattr(self.trainer.model, 'get_ckpt_name'):
            self.ckpt_prefix = self.trainer.model.get_ckpt_name() + '_'

        self.config['log_foldername'] = self.ckpt_foldername
        self.ckpt_foldername = os.path.join(self.save_loc,
                                            self.ckpt_foldername)
        self.pth_filepath = os.path.join(self.save_loc, self.ckpt_foldername,
                                         self.ckpt_prefix + "final.pth")

        self.models_foldername = os.path.join(self.ckpt_foldername, "models")
        if not os.path.exists(self.models_foldername):
            os.makedirs(self.models_foldername)

        self.save_config()

    def save_config(self):
        cfg_file = os.path.join(self.ckpt_foldername, "config.yaml")
        with open(cfg_file, 'w') as f:
            # Pop out config_override if present to remove clutter in
            # saved configuration yaml file
            self.config.pop('config_override', None)
            yaml.dump(self.config, f)

    def load_state_dict(self):
        if self.config['resume_file'] is not None and \
           os.path.exists(self.config['resume_file']):
            self._load(self.config['resume_file'])
            return

        ckpt_filepath = os.path.join(self.ckpt_foldername,
                                     self.ckpt_prefix + "best.ckpt")

        if self.config['resume'] is True \
           and os.path.exists(ckpt_filepath):
            self._load(ckpt_filepath)

    def _load(self, file):
        self.trainer.writer.write("Loading checkpoint")
        ckpt = self._torch_load(file)

        data_parallel = registry.get('data_parallel')

        ckpt_model = ckpt['model']

        pretrained_mapping = self.config['training_parameters'].get(
            'pretrained_mapping', {})

        if self.config['pretrained'] is None:
            pretrained_mapping = {}

        new_dict = {}

        # TODO: Move to separate function
        for attr in ckpt_model:
            if 'fa_history' in attr:
                new_dict[attr.replace('fa_history', 'fa_context')] = \
                    ckpt_model[attr]
            elif data_parallel is False:
                # In case the ckpt was actually a data parallel model
                # replace first module. from dataparallel with empty string
                new_dict[attr.replace("module.", "", 1)] = ckpt_model[attr]
            else:
                new_dict[attr] = ckpt_model[attr]

        if len(pretrained_mapping.items()) == 0:
            final_dict = new_dict

            self.trainer.model.load_state_dict(final_dict)
            self.trainer.optimizer.load_state_dict(ckpt['optimizer'])
            self.trainer.early_stopping.init_from_checkpoint(ckpt)

            self.trainer.writer.write("Checkpoint loaded")

            if 'best_iteration' in ckpt:
                self.trainer.current_iteration = ckpt['best_iteration']
                registry.register('current_iteration',
                                  self.trainer.current_iteration)

            if 'best_epoch' in ckpt:
                self.trainer.current_epoch = ckpt['best_epoch']
                registry.register('current_epoch',
                                  self.trainer.current_epoch)
        else:
            final_dict = {}
            model = self.trainer.model
            own_state = model.state_dict()
            for key, value in pretrained_mapping.items():
                key = "." + key + "."
                value = "." + value + "."
                for attr in new_dict:
                    for own_attr in own_state:
                        if 'fa_context' in own_attr:
                            continue
                        if key in attr and value in own_attr \
                            and attr.replace(key, "") == \
                                own_attr.replace(value, ""):
                            print("Copying", attr, own_attr)
                            own_state[own_attr].copy_(
                                new_dict[attr]
                            )
            self.trainer.writer.write("Pretrained model loaded")

    def _load_state_dict_mapping(self, ckpt_model):
        model = self.trainer.model
        attr_mapping = {
            'image_feature_encoders': 'img_feat_encoders',
            'image_feature_embeddings_list': 'img_embeddings_list',
            'image_text_multi_modal_combine_layer':
            'multi_modal_combine_layer',
            'text_embeddings': 'text_embeddings',
            'classifier': 'classifier'
        }

        data_parallel = registry.get('data_parallel')

        if not data_parallel:
            for key in attr_mapping:
                attr_mapping[key.replace("module.", "")] = attr_mapping[key]
                attr_mapping.pop(key)

        for key in attr_mapping:
            getattr(model, key).load_state_dict(ckpt_model[attr_mapping[key]])

    def _torch_load(self, file):
        if self.config['use_cuda']:
            return torch.load(file)
        else:
            return torch.load(file, map_location=lambda storage, loc: storage)

    def save(self, iteration, update_best=False):
        ckpt_filepath = os.path.join(self.models_foldername,
                                     "model_%d.ckpt" % iteration)
        best_ckpt_filepath = os.path.join(self.ckpt_foldername,
                                          self.ckpt_prefix + "best.ckpt")

        best_iteration = self.trainer.early_stopping.best_monitored_iteration
        best_metric = self.trainer.early_stopping.best_monitored_metric
        ckpt = {
            'model': self.trainer.model.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict(),
            'best_iteration': best_iteration,
            'best_metric': best_metric
        }

        torch.save(ckpt, ckpt_filepath)

        if update_best:
            torch.save(ckpt, best_ckpt_filepath)

    def restore(self):
        self.trainer.writer.write("Restoring checkpoint")
        best_path = os.path.join(self.ckpt_foldername,
                                 self.ckpt_prefix + "best.ckpt")
        if os.path.exists(best_path):
            ckpt = self._torch_load(best_path)
            self.trainer.model.load_state_dict(ckpt['model'])
            self.trainer.optimizer.load_state_dict(ckpt['optimizer'])

    def finalize(self):
        torch.save(self.trainer.model, self.pth_filepath)
