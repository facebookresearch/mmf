import yaml
import sys
import random
import torch
import os
import json


class Configuration:
    def __init__(self, config_yaml_file):
        self.config_path = config_yaml_file
        self.default_config = self._get_default_config_path()

        self.config = {}
        with open(self.default_config, 'r') as f:
            try:
                self.config = yaml.load(f)
            except yaml.YAMLError as err:
                print("[Error] Default config yaml error", err)

        if self.config_path is None:
            return

        user_config = {}
        with open(self.config_path, 'r') as f:
            try:
                # TODO: Create a default config here
                # and then update it with yaml config
                user_config = yaml.load(f)
            except yaml.YAMLError as err:
                print("Config yaml error", err)
                sys.exit(0)

        self.config.update(user_config)

    def get_config(self):
        return self.config

    def update_with_args(self, args):
        args_dict = vars(args)
        self._update_key(self.config, args_dict)
        self.config.update(args_dict)

        self._update_specific()

    def update_with_task_config(self, task_loader):
        task_loader.load_config()
        self.config.update(task_loader.task_config)

    def _update_key(self, dictionary, update_dict):
        for key, value in dictionary.items():
            if not isinstance(value, dict):
                if key in update_dict and update_dict[key] is not None:
                    dictionary[key] = update_dict[key]
            else:
                dictionary[key] = self._update_key(value, update_dict)

        return dictionary

    def pretty_print(self):
        print(json.dumps(self.config, indent=4, sort_keys=True))

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
            print("[Warning] CUDA option used but cuda is not present"
                  ". Switching to CPU version")
            self.config['use_cuda'] = False
