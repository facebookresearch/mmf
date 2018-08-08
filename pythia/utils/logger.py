import os

from tensorboardX import SummaryWriter

from pythia.utils.general import ckpt_name_from_core_args
from pythia.utils.timer import Timer


class Logger:
    def __init__(self, config):
        self.timer = Timer()

        self.config = config

        self.log_filename = ckpt_name_from_core_args(config) + '_'
        time_format = "%Y-%m-%dT%H:%M:%"
        self.log_filename += self.timer.get_time_hhmmss(None, time_format)
        self.log_filename += ".log"

        self.log_file = None
        self.summary_writer = None

        if 'log_dir' not in self.config:
            print("[Warning] log_dir missing from config")
            return

        self.summary_writer = SummaryWriter(self.config['log_dir'])

        if not os.path.exists(self.config['log_dir']):
            os.makedirs(self.config['log_dir'])

        self.log_filename = os.path.join(self.config['log_dir'],
                                         self.log_filename)

        self.log_file = open(self.log_filename, 'a', 4)
        self.should_log = not self.config['should_not_log']
        self.config['should_log'] = self.should_log

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()
        if self.log_file is not None:
            self.summary_writer.close()

    def write(self, x):
        if self.should_log and self.log_file is not None:
            self.log_file.write(str(x) + '\n')

        print(str(x) + '\n')

    def add_scalars(self, scalar_dict, iteration):
        if self.summary_writer is None:
            return
        for key, val in scalar_dict.items():
            self.summary_writer.add_scalar(key, val, iteration)

    def add_histogram_for_model(self, model, iteration):
        if self.summary_writer is None:
            return
        for name, param in model.parameters():
            np_param = param.clone().cpu().data.numpy()
            self.summary_writer.add_histogram(name, np_param, iteration)
