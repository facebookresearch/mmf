import os
import logging

from tensorboardX import SummaryWriter

from pythia.utils.general import ckpt_name_from_core_args, \
                                 foldername_from_config_override
from pythia.utils.timer import Timer


class Logger:
    def __init__(self, config):
        self.timer = Timer()

        self.config = config

        self.log_folder = ckpt_name_from_core_args(config)
        self.log_folder += foldername_from_config_override(config)
        time_format = "%Y-%m-%dT%H:%M:%S"
        self.log_filename = ckpt_name_from_core_args(config) + '_'
        self.log_filename += self.timer.get_time_hhmmss(None, time_format)
        self.log_filename += ".log"

        self.log_file = None
        self.summary_writer = None

        tensboard_folder = os.path.join(self.log_folder, "tensorboard")
        self.summary_writer = SummaryWriter(tensboard_folder)

        self.log_filename = os.path.join(self.log_folder,
                                         self.log_filename)

        logging.basicConfig(filename=self.log_filename,level=logging.DEBUG,
                            filemode='a', format="%(levelname)s: %(message)s")
        self.logger = logging.getLogger(__name__)
        self.should_log = not self.config['should_not_log']
        self.config['should_log'] = self.should_log

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()
        if self.log_file is not None:
            self.summary_writer.close()

    def write(self, x, level='debug'):
        # if it should not log then just print it
        if self.should_log and self.log_file is not None:
            if hasattr(self.logger, level):
                getattr(self.logger, level)(str(x))
            else:
                self.logger.error("Unknown log level type: %s" % level)
        else:
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
