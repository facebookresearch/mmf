import yaml
import sys


class Configuration:
    def __init__(self, config_yaml_file):
        self.config_path = config_yaml_file

        with open(self.config_path, 'r') as f:
            try:
                self.config = yaml.load(f)
            except yaml.YAMLError as err:
                print('Config yaml error', err)
                sys.exit(0)

    def get_config(self):
        return self.config

    def update_with_args(self, args):
        args_dict = vars(args)
        self._update_key(self.config, args_dict)
        self.config.update(args_dict)

    def _update_key(self, dictionary, update_dict):
        for key, value in dictionary.items():
            if not isinstance(value, dict):
                if key in update_dict:
                    dictionary[key] = update_dict[key]
            else:
                dictionary[key] = self._update_key(value, update_dict)

        return dictionary
