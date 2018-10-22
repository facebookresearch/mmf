import os
import torch
import yaml

from pythia.utils.general import ckpt_name_from_core_args, \
                                 foldername_from_config_override


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

        ckpt_filepath = os.path.join(self.ckpt_foldername,
                                     self.ckpt_prefix + "best.ckpt")

        if self.config['resume'] is True \
           and os.path.exists(ckpt_filepath):
            self._load(ckpt_filepath)

    def _load(self, file):
        self.trainer.writer.write("Loading checkpoint")
        ckpt = self._torch_load(file)

        self.trainer.model.load_state_dict(ckpt['model'])
        self.trainer.optimizer.load_state_dict(ckpt['optimizer'])
        self.trainer.early_stopping.init_from_checkpoint(ckpt)

        self.trainer.writer.write("Checkpoint loaded")

        if 'best_iteration' in ckpt:
            self.trainer.current_iteration = ckpt['best_iteration']

        if 'best_epoch' in ckpt:
            self.trainer.current_epoch = ckpt['best_epoch']

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
        best_path = os.path.join(self.ckpt_foldername,
                                 self.ckpt_prefix + "best.ckpt")
        if os.path.exists(best_path):
            ckpt = self._torch_load(best_path)
            self.trainer.model.load_state_dict(ckpt['model'])
            self.trainer.optimizer.load_state_dict()

    def finalize(self):
        torch.save(self.trainer.model, self.pth_filepath)
