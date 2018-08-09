import os
import torch

from pythia.utils.general import ckpt_name_from_core_args


class Checkpoint:
    def __init__(self, trainer):
        '''
        Generates a path for saving model which can also be used for resuming
        from a checkpoint.
        '''
        self.trainer = trainer

        self.config = self.trainer.config
        self.save_loc = self.config['save_loc']

        self.ckpt_filename = ckpt_name_from_core_args(self.config)

        if hasattr(self.trainer.model, 'get_ckpt_name'):
            self.ckpt_filename += self.trainer.model.get_ckpt_name()

        self.config['log_filename'] = self.ckpt_filename
        self.ckpt_filepath = os.path.join(self.save_loc, self.ckpt_filename
                                          + ".ckpt")
        self.pth_filepath = os.path.join(self.save_loc, self.ckpt_filename
                                         + ".pth")

    def load_state_dict(self):
        if self.config['resume_file'] is not None and \
           os.path.exists(self.config['resume_file']):
            self._load(self.config['resume_file'])

        if self.config['resume'] is True \
           and os.path.exists(self.ckpt_filepath):
            self._load(self.config['resume'])

    def _load(self, file):
        self.trainer.writer.write("Loading checkpoint")
        ckpt = self._torch_load(file)

        self.trainer.model.load_state_dict(ckpt['model'])
        self.trainer.optimizer.load_state_dict(ckpt['optimizer'])

    def _torch_load(self, file):
        if self.config['use_gpu']:
            return torch.load(file)
        else:
            return torch.load(file, map_location=lambda storage, loc: storage)

    def save(self):
        if not os.path.exists(os.path.dirname(self.ckpt_filepath)):
            os.mkdir(os.path.dirname(self.ckpt_filepath))

        ckpt = {
            'model': self.trainer.model.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict()
        }

        torch.save(ckpt, self.ckpt_filepath)

    def restore(self):
        if os.path.exists(self.ckpt_filepath):
            self.trainer.model.load_state_dict(
                self._torch_load(self.ckpt_filepath))

    def finalize(self):
        if not os.path.exists(os.path.dirname(self.pth_filepath)):
            os.mkdir(os.path.dirname(self.pth_filepath))

        torch.save(self.trainer.model, self.pth_filepath)
