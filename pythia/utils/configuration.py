# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import json
import os
import random
import warnings
from ast import literal_eval

import demjson
import torch
import yaml
from omegaconf import OmegaConf

from pythia.common.registry import registry
from pythia.utils.general import get_pythia_root


def load_yaml(f):
    mapping = OmegaConf.load(f)

    if mapping is None:
        mapping = OmegaConf.create()

    includes = mapping.get("includes", [])

    if not isinstance(includes, collections.abc.Sequence):
        raise AttributeError(
            "Includes must be a list, {} provided".format(type(includes))
        )

    include_mapping = OmegaConf.create()

    pythia_root_dir = get_pythia_root()

    for include in includes:
        include = os.path.join(pythia_root_dir, include)
        current_include_mapping = load_yaml(include)
        include_mapping = OmegaConf.merge(include_mapping, current_include_mapping)

    mapping.pop("includes", None)

    mapping = OmegaConf.merge(include_mapping, mapping)

    return mapping


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
    def __init__(self, args):
        self.config = {}
        self.args = args
        self._register_resolvers()

        default_config = self._build_default_config()
        opts_config = self._build_opt_list(args.opts)
        user_config = self._build_user_config(opts_config)
        model_config = self._build_model_config(opts_config)
        dataset_config = self._build_dataset_config(opts_config)
        args_overrides = self._build_args_overrides(args)

        self._default_config = default_config
        self._user_config = user_config
        self.config = OmegaConf.merge(
            default_config, model_config, dataset_config, user_config, args_overrides
        )

        # TODO: Remove in next iteration
        self.config = self._update_with_args(self.config, args)

        self.config = self._merge_with_dotlist(self.config, args.opts)
        self._update_specific(self.config)

    def _build_default_config(self):
        self.default_config_path = self._get_default_config_path()
        default_config = load_yaml(self.default_config_path)
        return default_config

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    def _build_user_config(self, opts):
        user_config = {}

        # Update user_config with opts if passed
        self.config_path = opts.config
        if self.config_path is not None:
            user_config = load_yaml(self.config_path)

        return user_config

    def _build_args_overrides(self, args):
        # Update with demjson if passed
        demjson_config = self._get_demjson_config(args.config_override)
        # TODO: Remove in next iteration
        args_config = self._get_args_config(args)
        return OmegaConf.merge(demjson_config, args_config)

    def _build_model_config(self, config):
        model = config.model
        if model is None:
            raise KeyError("Required argument 'model' not passed")
        model_cls = registry.get_model_class(model)

        if model_cls is None:
            warnings.warn("No model named '{}' has been registered".format(model))
            return OmegaConf.create()

        default_model_config_path = model_cls.config_path()

        if default_model_config_path is None:
            warnings.warn(
                "Model {}'s class has no default configuration provided".format(model)
            )
            return OmegaConf.create()

        return load_yaml(default_model_config_path)

    def _build_dataset_config(self, config):
        dataset = config.dataset
        datasets = config.datasets

        if dataset is None and datasets is None:
            raise KeyError("Required argument 'dataset|datasets' not passed")

        if datasets is None:
            config.datasets = dataset
            datasets = dataset.split(",")
        else:
            datasets = datasets.split(",")

        dataset_config = OmegaConf.create()

        for dataset in datasets:
            builder_cls = registry.get_builder_class(dataset)

            if builder_cls is None:
                warnings.warn(
                    "No dataset named '{}' has been registered".format(dataset)
                )
                continue
            default_dataset_config_path = builder_cls.config_path()
            if default_dataset_config_path is None:
                warnings.warn(
                    "Dataset {}'s builder class has no default configuration provided".format(
                        dataset
                    )
                )
                continue
            dataset_config = OmegaConf.merge(
                dataset_config, load_yaml(default_dataset_config_path)
            )

        return dataset_config

    def get_config(self):
        self._register_resolvers()
        return self.config

    def _update_with_args(self, config, args, force=False):
        args_dict = vars(args)

        self._update_key(config, args_dict)
        if force is True:
            config.update(args_dict)

        return config

    def _get_demjson_config(self, demjson_string):
        if demjson_string is None:
            return OmegaConf.create()

        demjson_dict = demjson.decode(demjson_string)
        return OmegaConf.create(demjson_dict)

    def _get_args_config(self, args):
        args_dict = vars(args)
        return OmegaConf.create(args_dict)

    def _register_resolvers(self):
        OmegaConf.clear_resolvers()
        # Device count resolver
        device_count = max(1, torch.cuda.device_count())
        OmegaConf.register_resolver("device_count", lambda: device_count)

    def _merge_with_dotlist(self, config, opts):
        # TODO: To remove technical debt, a possible solution is to use
        # struct mode to update with dotlist OmegaConf node. Look into this
        # in next iteration
        if opts is None:
            opts = []

        if len(opts) == 0:
            return

        # Support equal e.g. model=visual_bert for better future hydra support
        has_equal = opts[0].find("=") != -1

        if has_equal:
            opt_values = [opt.split("=") for opt in opts]
        else:
            assert len(opts) % 2 == 0, "Number of opts should be multiple of 2"
            opt_values = zip(opts[0::2], opts[1::2])

        for opt, value in opt_values:
            if opt == "dataset":
                opt = "datasets"

            splits = opt.split(".")
            current = config
            for idx, field in enumerate(splits):
                array_index = -1
                if field.find("[") != -1 and field.find("]") != -1:
                    stripped_field = field[: field.find("[")]
                    array_index = int(field[field.find("[") + 1 : field.find("]")])
                else:
                    stripped_field = field
                if stripped_field not in current:
                    raise AttributeError(
                        "While updating configuration"
                        " option {} is missing from"
                        " configuration at field {}".format(opt, stripped_field)
                    )
                if isinstance(current[stripped_field], collections.abc.Mapping):
                    current = current[stripped_field]
                elif (
                    isinstance(current[stripped_field], collections.abc.Sequence)
                    and array_index != -1
                ):
                    current_value = current[stripped_field][array_index]

                    # Case where array element to be updated is last element
                    if not isinstance(
                        current_value,
                        (collections.abc.Mapping, collections.abc.Sequence),
                    ):
                        print("Overriding option {} to {}".format(opt, value))
                        current[stripped_field][array_index] = self._decode_value(value)
                    else:
                        # Otherwise move on down the chain
                        current = current_value
                else:
                    if idx == len(splits) - 1:
                        print("Overriding option {} to {}".format(opt, value))
                        current[stripped_field] = self._decode_value(value)
                    else:
                        raise AttributeError(
                            "While updating configuration",
                            "option {} is not present "
                            "after field {}".format(opt, stripped_field),
                        )

        return config

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

    def freeze(self):
        # self.config = ConfigNode(self.config)
        OmegaConf.set_struct(self.config, True)

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        # Support equal e.g. model=visual_bert for better future hydra support
        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def pretty_print(self):
        if not self.config.training_parameters.log_detailed_config:
            return

        self.writer = registry.get("writer")

        self.writer.write("=====  Training Parameters    =====", "info")
        self.writer.write(
            self._convert_node_to_json(self.config.training_parameters), "info"
        )

        self.writer.write("======  Dataset Attributes  ======", "info")
        datasets = self.config.datasets.split(",")

        for dataset in datasets:
            if dataset in self.config.dataset_attributes:
                self.writer.write("======== {} =======".format(dataset), "info")
                dataset_config = self.config.dataset_attributes[dataset]
                self.writer.write(self._convert_node_to_json(dataset_config), "info")
            else:
                self.writer.write(
                    "No dataset named '{}' in config. Skipping".format(dataset),
                    "warning",
                )

        self.writer.write("======  Optimizer Attributes  ======", "info")
        self.writer.write(
            self._convert_node_to_json(self.config.optimizer_attributes), "info"
        )

        if self.config.model not in self.config.model_attributes:
            raise ValueError(
                "{} not present in model attributes".format(self.config.model)
            )

        self.writer.write(
            "======  Model ({}) Attributes  ======".format(self.config.model), "info"
        )
        self.writer.write(
            self._convert_node_to_json(self.config.model_attributes[self.config.model]),
            "info",
        )

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def _get_default_config_path(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(
            directory, "..", "common", "defaults", "configs", "base.yml"
        )

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

    def _update_specific(self, config):
        self.writer = registry.get("writer")
        tp = self.config.training_parameters

        # if args["seed"] is not None or tp['seed'] is not None:
        #     print(
        #         "You have chosen to seed the training. This will turn on CUDNN deterministic "
        #         "setting which can slow down your training considerably! You may see unexpected "
        #         "behavior when restarting from checkpoints."
        #     )

        # if args["seed"] == -1:
        #     self.config["training_parameters"]["seed"] = random.randint(1, 1000000)

        if config.learning_rate:
            if "optimizer" in config and "params" in config.optimizer:
                lr = config.learning_rate
                config.optimizer_attributes.params.lr = lr

        if (
            not torch.cuda.is_available()
            and "cuda" in config.training_parameters.device
        ):
            warnings.warn(
                "Device specified is 'cuda' but cuda is not present. Switching to CPU version"
            )
            config.training_parameters.device = "cpu"

        return config
