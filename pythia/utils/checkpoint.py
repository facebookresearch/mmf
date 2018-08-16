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
        self.ckpt_filename = os.path.join(self.save_loc, self.ckpt_filename)
        self.pth_filepath = os.path.join(self.save_loc, self.ckpt_filename
                                         + ".pth")

    def load_state_dict(self):
        if self.config['resume_file'] is not None and \
           os.path.exists(self.config['resume_file']):
            self._load(self.config['resume_file'])

        if self.config['resume'] is True \
           and os.path.exists(self.ckpt_filepath):
            self._load(self.ckpt_filepath)

    def _load(self, file):
        self.trainer.writer.write("Loading checkpoint")
        ckpt = self._torch_load(file)

        self.trainer.model.load_state_dict(ckpt['model'])
        self.trainer.optimizer.load_state_dict(ckpt['optimizer'])
        self.trainer.early_stopping.init_from_checkpoint(ckpt)

    def _torch_load(self, file):
        if self.config['use_cuda']:
            return torch.load(file)
        else:
            return torch.load(file, map_location=lambda storage, loc: storage)

    def save(self, iteration, update_best=False):
        ckpt_filepath = self.ckpt_filename + "_%d.ckpt" % iteration
        best_ckpt_filepath = self.ckpt_filename + "_best.ckpt"

        if not os.path.exists(os.path.dirname(ckpt_filepath)):
            os.mkdir(os.path.dirname(ckpt_filepath))

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
        best_path = self.ckpt_filepath + "_best.ckpt"
        if os.path.exists(best_path):
            self.trainer.model.load_state_dict(
                self._torch_load(best_path))

    def finalize(self):
        if not os.path.exists(os.path.dirname(self.pth_filepath)):
            os.mkdir(os.path.dirname(self.pth_filepath))

        torch.save(self.trainer.model, self.pth_filepath)
