# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import json
import os
import random
from ast import literal_eval

import yaml

import demjson
import torch
from pythia.common.registry import registry
from pythia.utils.general import get_pythia_root
from pythia.utils.distributed_utils import is_main_process


class ConfigNode(collections.OrderedDict):
    IMMUTABLE = "__is_frozen"

    def __init__(self, init_dict={}):
        self.__dict__[ConfigNode.IMMUTABLE] = False
        super().__init__(init_dict)

        for key in self:
            if isinstance(self[key], collections.abc.Mapping):
                self[key] = ConfigNode(self[key])
            elif isinstance(self[key], list):
                for idx, item in enumerate(self[key]):
                    if isinstance(item, collections.abc.Mapping):
                        self[key][idx] = ConfigNode(item)

    def freeze(self):
        for field in self.keys():
            if isinstance(self[field], collections.abc.Mapping):
                self[field].freeze()
            elif isinstance(self[field], list):
                for item in self[field]:
                    if isinstance(item, collections.abc.Mapping):
                        item.freeze()

        self.__dict__[ConfigNode.IMMUTABLE] = True

    def defrost(self):
        for field in self.keys():
            if isinstance(self[field], collections.abc.Mapping):
                self[field].defrost()
            elif isinstance(self[field], list):
                for item in self[field]:
                    if isinstance(item, collections.abc.Mapping):
                        item.defrost()

        self.__dict__[ConfigNode.IMMUTABLE] = False

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)

        return self[key]

    def __setattr__(self, key, value):
        if self.__dict__[ConfigNode.IMMUTABLE] is True:
            raise AttributeError("ConfigNode has been frozen and can't be updated")

        self[key] = value

    def _indent(self, st, num_spaces):
        st = st.split("\n")
        first = st.pop(0)
        st = [(num_spaces * " ") + line for line in st]
        st = [first] + st
        st = "\n".join(st)
        return st

    def __str__(self):
        strs = []

        if isinstance(self, collections.abc.Mapping):
            for key, value in sorted(self.items()):
                seperator = "\n" if isinstance(value, ConfigNode) else " "
                if isinstance(value, list):
                    attr_str = ["{}:".format(key)]
                    for item in value:
                        item_str = self._indent(str(item), 2)
                        attr_str.append("- {}".format(item_str))
                    attr_str = "\n".join(attr_str)
                else:
                    attr_str = "{}:{}{}".format(str(key), seperator, str(value))
                    attr_str = self._indent(attr_str, 2)
                strs.append(attr_str)
        return "\n".join(strs)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super().__repr__())


class Configuration:
    def __init__(self, config_yaml_file):
        self.config_path = config_yaml_file
        self.default_config = self._get_default_config_path()
        self.config = {}

        base_config = {}

        base_config = self.load_yaml(self.default_config)

        user_config = {}

        if self.config_path is not None:
            user_config = self.load_yaml(self.config_path)

        self._base_config = base_config
        self._user_config = user_config

        self.config = self.nested_dict_update(base_config, user_config)

    def get_config(self):
        return self.config

    def load_yaml(self, file):
        with open(file, "r") as stream:
            mapping = yaml.safe_load(stream)

            if mapping is None:
                mapping = {}

            includes = mapping.get("includes", [])

            if not isinstance(includes, list):
                raise AttributeError(
                    "Includes must be a list, {} provided".format(type(includes))
                )
            include_mapping = {}

            pythia_root_dir = get_pythia_root()

            for include in includes:
                include = os.path.join(pythia_root_dir, include)
                current_include_mapping = self.load_yaml(include)
                include_mapping = self.nested_dict_update(
                    include_mapping, current_include_mapping
                )

            mapping.pop("includes", None)

            mapping = self.nested_dict_update(include_mapping, mapping)

            return mapping

    def update_with_args(self, args, force=False):
        args_dict = vars(args)

        self._update_key(self.config, args_dict)
        if force is True:
            self.config.update(args_dict)
        self._update_specific(args_dict)

    def override_with_cmd_config(self, cmd_config):
        if cmd_config is None:
            return

        cmd_config = demjson.decode(cmd_config)
        self.config = self.nested_dict_update(self.config, cmd_config)

    def nested_dict_update(self, dictionary, update):
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
        if dictionary is None:
            dictionary = {}

        for k, v in update.items():
            if isinstance(v, collections.abc.Mapping):
                dictionary[k] = self.nested_dict_update(dictionary.get(k, {}), v)
            else:
                dictionary[k] = self._decode_value(v)
        return dictionary

    def freeze(self):
        self.config = ConfigNode(self.config)
        self.config.freeze()

    def _merge_from_list(self, opts):
        if opts is None:
            opts = []

        assert len(opts) % 2 == 0, "Number of opts should be multiple of 2"

        for opt, value in zip(opts[0::2], opts[1::2]):
            splits = opt.split(".")
            current = self.config
            for idx, field in enumerate(splits):
                if field not in current:
                    raise AttributeError(
                        "While updating configuration"
                        " option {} is missing from"
                        " configuration at field {}".format(opt, field)
                    )
                if not isinstance(current[field], collections.abc.Mapping):
                    if idx == len(splits) - 1:
                        if is_main_process():
                            print("Overriding option {} to {}".format(opt, value))

                        current[field] = self._decode_value(value)
                    else:
                        raise AttributeError(
                            "While updating configuration",
                            "option {} is not present "
                            "after field {}".format(opt, field),
                        )
                else:
                    current = current[field]

    def override_with_cmd_opts(self, opts):
        self._merge_from_list(opts)

    def _decode_value(self, value):
        # https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L400
        if not isinstance(value, str):
            return value

        if value == "None":
            value = None

        try:
            value = literal_eval(value)
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value

    def _update_key(self, dictionary, update_dict):
        """
        Takes a single depth dictionary update_dict and uses it to
        update 'dictionary' whenever key in 'update_dict' is found at
        any level in 'dictionary'
        """
        for key, value in dictionary.items():
            if not isinstance(value, collections.abc.Mapping):
                if key in update_dict and update_dict[key] is not None:
                    dictionary[key] = update_dict[key]
            else:
                dictionary[key] = self._update_key(value, update_dict)

        return dictionary

    def pretty_print(self):
        self.writer = registry.get("writer")

        self.writer.write("=====  Training Parameters    =====", "info")
        self.writer.write(
            json.dumps(self.config.training_parameters, indent=4, sort_keys=True),
            "info",
        )

        self.writer.write("======  Task Attributes  ======", "info")
        tasks = self.config.tasks.split(",")
        datasets = self.config.datasets.split(",")

        for task in tasks:
            if task not in self.config.task_attributes:
                raise ValueError(
                    "Task {} not present in task_attributes config".format(task)
                )

            task_config = self.config.task_attributes[task]

            for dataset in datasets:
                if dataset in task_config.dataset_attributes:
                    self.writer.write(
                        "======== {}/{} =======".format(task, dataset), "info"
                    )
                    dataset_config = task_config.dataset_attributes[dataset]
                    self.writer.write(
                        json.dumps(dataset_config, indent=4, sort_keys=True), "info"
                    )

        self.writer.write("======  Optimizer Attributes  ======", "info")
        self.writer.write(
            json.dumps(self.config.optimizer_attributes, indent=4, sort_keys=True),
            "info",
        )

        if self.config.model not in self.config.model_attributes:
            raise ValueError(
                "{} not present in model attributes".format(self.config.model)
            )

        self.writer.write(
            "======  Model ({}) Attributes  ======".format(self.config.model), "info"
        )
        self.writer.write(
            json.dumps(
                self.config.model_attributes[self.config.model],
                indent=4,
                sort_keys=True,
            ),
            "info",
        )

    def _get_default_config_path(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(
            directory, "..", "common", "defaults", "configs", "base.yml"
        )

    def _update_specific(self, args):
        self.writer = registry.get("writer")
        tp = self.config["training_parameters"]

        if args["seed"] is not None or tp['seed'] is not None:
            print(
                "You have chosen to seed the training. This will turn on CUDNN deterministic "
                "setting which can slow down your training considerably! You may see unexpected "
                "behavior when restarting from checkpoints."
            )

        if args["seed"] == -1:
            self.config["training_parameters"]["seed"] = random.randint(1, 1000000)

        if "learning_rate" in args:
            if "optimizer" in self.config and "params" in self.config["optimizer"]:
                lr = args["learning_rate"]
                self.config["optimizer_attributes"]["params"]["lr"] = lr

        if (
            not torch.cuda.is_available()
            and "cuda" in self.config["training_parameters"]["device"]
        ):
            if is_main_process():
                print(
                    "WARNING: Device specified is 'cuda' but cuda is "
                    "not present. Switching to CPU version"
                )
            self.config["training_parameters"]["device"] = "cpu"

        if tp["distributed"] is True and tp["data_parallel"] is True:
            print(
                "training_parameters.distributed and "
                "training_parameters.data_parallel are "
                "mutually exclusive. Setting "
                "training_parameters.distributed to False"
            )
            tp["distributed"] = False
