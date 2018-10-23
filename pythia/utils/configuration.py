import yaml
import sys
import random
import torch
import os
import json
import demjson
import collections

from .general import nested_dict_update
from pythia.core.registry import Registry


class Configuration:
    def __init__(self, config_yaml_file):
        self.config_path = config_yaml_file
        self.default_config = self._get_default_config_path()

        self.config = {}
        with open(self.default_config, 'r') as f:
            try:
                self.config = yaml.load(f)
            except yaml.YAMLError as err:
                self.writer.write("Default config yaml error: " + err,
                                  'error')

        user_config = {}
        self.user_config = user_config

        if self.config_path is None:
            return

        with open(self.config_path, 'r') as f:
            try:
                # TODO: Create a default config here
                # and then update it with yaml config
                user_config = yaml.load(f)
            except yaml.YAMLError as err:
                self.writer.write("Config yaml error: " + err, 'error')
                sys.exit(0)
        self.user_config = user_config

    def get_config(self):
        return self.config

    def update_with_args(self, args, force=False):
        args_dict = vars(args)

        self._update_key(self.config, args_dict)
        if force is True:
            self.config.update(args_dict)
        self._update_specific()

    def update_with_task_config(self, config):
        self.config = nested_dict_update(self.config, config)
        # At this point update with user's config
        self._update_with_user_config()

    def _update_with_user_config(self):
        self.config = nested_dict_update(self.config, self.user_config)

    def override_with_cmd_config(self, cmd_config):
        if cmd_config is None:
            return

        cmd_config = demjson.decode(cmd_config)
        self.config = nested_dict_update(self.config, cmd_config)

    def _update_key(self, dictionary, update_dict):
        '''
        Takes a single depth dictionary update_dict and uses it to
        update 'dictionary' whenever key in 'update_dict' is found at
        any level in 'dictionary'
        '''
        for key, value in dictionary.items():
            if not isinstance(value, collections.Mapping):
                if key in update_dict and update_dict[key] is not None:
                    dictionary[key] = update_dict[key]
            else:
                dictionary[key] = self._update_key(value, update_dict)

        return dictionary

    def pretty_print(self):
        self.writer = Registry.get('writer')

        self.writer.write("=====  Training Parameters    =====", "info")
        self.writer.write(json.dumps(self.config['training_parameters'],
                                     indent=4, sort_keys=True), "info")

        self.writer.write("======  Task Attributes  ======", "info")
        self.writer.write(json.dumps(self.config['task_attributes'],
                                     indent=4, sort_keys=True), "info")

        self.writer.write("======  Optimizer Attributes  ======", "info")
        self.writer.write(json.dumps(self.config['optimizer'],
                                     indent=4, sort_keys=True), "info")

        self.writer.write("======  Model Attributes  ======", "info")
        self.writer.write(json.dumps(self.config['model_attributes'],
                                     indent=4, sort_keys=True), "info")

    def _get_default_config_path(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(directory, '..', 'config', 'default.yml')

    def _update_specific(self):
        if self.config['seed'] <= 0:
            self.config['seed'] = random.randint(1, 1000000)

        if 'learning_rate' in self.config:
            if 'optimizer' in self.config and \
               'params' in self.config['optimizer']:
                lr = self.config['learning_rate']
                self.config['optimizer']['params']['lr'] = lr

        if not torch.cuda.is_available() or self.config['no_cuda'] is True:
            self.writer.write("CUDA option used but cuda is not present"
                              ". Switching to CPU version", 'warning')
            self.config['use_cuda'] = False
