# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import json
import os
import warnings
from ast import literal_eval

import demjson
import torch
from omegaconf import OmegaConf

from mmf.common.registry import registry
from mmf.utils.general import get_mmf_root


def load_yaml(f):
    # Convert to absolute path for loading includes
    f = os.path.abspath(f)
    mapping = OmegaConf.load(f)

    if mapping is None:
        mapping = OmegaConf.create()

    includes = mapping.get("includes", [])

    if not isinstance(includes, collections.abc.Sequence):
        raise AttributeError(
            "Includes must be a list, {} provided".format(type(includes))
        )

    include_mapping = OmegaConf.create()

    pythia_root_dir = get_mmf_root()

    for include in includes:
        original_include_path = include
        include = os.path.join(pythia_root_dir, include)

        # If path doesn't exist relative to MMF root, try relative to current file
        if not os.path.exists(include):
            include = os.path.join(os.path.dirname(f), original_include_path)

        current_include_mapping = load_yaml(include)
        include_mapping = OmegaConf.merge(include_mapping, current_include_mapping)

    mapping.pop("includes", None)

    mapping = OmegaConf.merge(include_mapping, mapping)

    return mapping


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
        args_overrides = self._build_demjson_config(args.config_override)

        self._default_config = default_config
        self._user_config = user_config
        self.config = OmegaConf.merge(
            default_config, model_config, dataset_config, user_config, args_overrides
        )

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

    def _build_model_config(self, config):
        model = config.model
        if model is None:
            raise KeyError("Required argument 'model' not passed")
        model_cls = registry.get_model_class(model)

        if model_cls is None:
            warning = "No model named '{}' has been registered".format(model)
            warnings.warn(warning)
            return OmegaConf.create()

        default_model_config_path = model_cls.config_path()

        if default_model_config_path is None:
            warning = "Model {}'s class has no default configuration provided".format(
                model
            )
            warnings.warn(warning)
            return OmegaConf.create()

        default_model_config_path = os.path.join(
            get_mmf_root(), default_model_config_path
        )
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
                warning = "No dataset named '{}' has been registered".format(dataset)
                warnings.warn(warning)
                continue
            default_dataset_config_path = builder_cls.config_path()
            if default_dataset_config_path is None:
                warning = "Dataset {}'s builder class has no default configuration provided".format(
                    dataset
                )
                warnings.warn(warning)
                continue

            default_dataset_config_path = os.path.join(
                get_mmf_root(), default_dataset_config_path
            )
            dataset_config = OmegaConf.merge(
                dataset_config, load_yaml(default_dataset_config_path)
            )

        return dataset_config

    def get_config(self):
        self._register_resolvers()
        return self.config

    def _build_demjson_config(self, demjson_string):
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
        OmegaConf.set_struct(self.config, True)

    def defrost(self):
        OmegaConf.set_struct(self.config, False)

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
        if not self.config.training.log_detailed_config:
            return

        self.writer = registry.get("writer")

        self.writer.write("=====  Training Parameters    =====", "info")
        self.writer.write(self._convert_node_to_json(self.config.training), "info")

        self.writer.write("======  Dataset Attributes  ======", "info")
        datasets = self.config.datasets.split(",")

        for dataset in datasets:
            if dataset in self.config.dataset_config:
                self.writer.write("======== {} =======".format(dataset), "info")
                dataset_config = self.config.dataset_config[dataset]
                self.writer.write(self._convert_node_to_json(dataset_config), "info")
            else:
                self.writer.write(
                    "No dataset named '{}' in config. Skipping".format(dataset),
                    "warning",
                )

        self.writer.write("======  Optimizer Attributes  ======", "info")
        self.writer.write(self._convert_node_to_json(self.config.optimizer), "info")

        if self.config.model not in self.config.model_config:
            raise ValueError(
                "{} not present in model attributes".format(self.config.model)
            )

        self.writer.write(
            "======  Model ({}) Attributes  ======".format(self.config.model), "info"
        )
        self.writer.write(
            self._convert_node_to_json(self.config.model_config[self.config.model]),
            "info",
        )

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def _get_default_config_path(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(directory, "..", "configs", "defaults.yaml")

    def _update_specific(self, config):
        self.writer = registry.get("writer")
        # tp = self.config.training

        # if args["seed"] is not None or tp['seed'] is not None:
        #     print(
        #         "You have chosen to seed the training. This will turn on CUDNN deterministic "
        #         "setting which can slow down your training considerably! You may see unexpected "
        #         "behavior when restarting from checkpoints."
        #     )

        # if args["seed"] == -1:
        #     self.config["training"]["seed"] = random.randint(1, 1000000)

        if config.learning_rate:
            if "optimizer" in config and "params" in config.optimizer:
                lr = config.learning_rate
                config.optimizer.params.lr = lr

        if not torch.cuda.is_available() and "cuda" in config.training.device:
            warnings.warn(
                "Device specified is 'cuda' but cuda is not present. Switching to CPU version"
            )
            config.training.device = "cpu"

        return config


# This is still here due to legacy reasons around
# older checkpoint loading from v0.3
class ConfigNode(collections.OrderedDict):
    pass
