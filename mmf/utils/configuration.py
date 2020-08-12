# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import json
import logging
import os
import warnings
from ast import literal_eval

import demjson
import torch
from mmf.common.registry import registry
from mmf.utils.env import import_user_module
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_absolute_path, get_mmf_root
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


def load_yaml(f):
    # Convert to absolute path for loading includes
    abs_f = get_absolute_path(f)

    try:
        mapping = OmegaConf.load(abs_f)
        f = abs_f
    except FileNotFoundError as e:
        # Check if this file might be relative to root?
        # TODO: Later test if this can be removed
        relative = os.path.abspath(os.path.join(get_mmf_root(), f))
        if not PathManager.isfile(relative):
            raise e
        else:
            f = relative
            mapping = OmegaConf.load(f)

    if mapping is None:
        mapping = OmegaConf.create()

    includes = mapping.get("includes", [])

    if not isinstance(includes, collections.abc.Sequence):
        raise AttributeError(
            "Includes must be a list, {} provided".format(type(includes))
        )

    include_mapping = OmegaConf.create()

    mmf_root_dir = get_mmf_root()

    for include in includes:
        original_include_path = include
        include = os.path.join(mmf_root_dir, include)

        # If path doesn't exist relative to MMF root, try relative to current file
        if not PathManager.exists(include):
            include = os.path.join(os.path.dirname(f), original_include_path)

        current_include_mapping = load_yaml(include)
        include_mapping = OmegaConf.merge(include_mapping, current_include_mapping)

    mapping.pop("includes", None)

    mapping = OmegaConf.merge(include_mapping, mapping)

    return mapping


def get_default_config_path():
    directory = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(directory, "..", "configs")

    # Check for fb defaults
    fb_defaults = os.path.join(configs_dir, "fb_defaults.yaml")
    if PathManager.exists(fb_defaults):
        return fb_defaults
    else:
        return os.path.join(configs_dir, "defaults.yaml")


def load_yaml_with_defaults(f):
    default_config = get_default_config_path()
    return OmegaConf.merge(load_yaml(default_config), load_yaml(f))


def get_zoo_config(
    key, variation="defaults", zoo_config_path=None, zoo_type="datasets"
):
    version = None
    resources = None
    if zoo_config_path is None:
        zoo_config_path = os.path.join("configs", "zoo", f"{zoo_type}.yaml")
    zoo = load_yaml(zoo_config_path)

    # Set struct on zoo so that unidentified access is not allowed
    OmegaConf.set_struct(zoo, True)

    try:
        item = OmegaConf.select(zoo, key)
    except Exception:
        # Key wasn't present or something else happened, return None, None
        return version, resources

    if not item:
        return version, resources

    if variation not in item:
        # If variation is not present, then key value should
        # be directly returned if "defaults" was selected as the variation
        assert (
            variation == "defaults"
        ), f"'{variation}' variation not present in zoo config"
        return _get_version_and_resources(item)
    elif "resources" in item:
        # Case where full key is directly passed
        return _get_version_and_resources(item)
    else:
        return _get_version_and_resources(item[variation])


def _get_version_and_resources(item):
    assert "version" in item, "'version' key should be present in zoo config {}".format(
        item._get_full_key("")
    )
    assert (
        "resources" in item
    ), "'resources' key should be present in zoo config {}".format(
        item._get_full_key("")
    )

    return item.version, item.resources


def get_global_config(key=None):
    config = registry.get("config")
    if config is None:
        configuration = Configuration()
        config = configuration.get_config()
        registry.register("config", config)

    if key:
        config = OmegaConf.select(config, key)

    return config


def get_mmf_cache_dir():
    config = get_global_config()
    cache_dir = config.env.cache_dir
    # If cache_dir path exists do not join to mmf root
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(get_mmf_root(), cache_dir)
    return cache_dir


def get_mmf_env(key=None):
    config = get_global_config()
    if key:
        return OmegaConf.select(config.env, key)
    else:
        return config.env


def resolve_cache_dir(env_variable="MMF_CACHE_DIR", default="mmf"):
    # Some of this follow what "transformers" does for there cache resolving
    try:
        from torch.hub import _get_torch_home

        torch_cache_home = _get_torch_home()
    except ImportError:
        torch_cache_home = os.path.expanduser(
            os.getenv(
                "TORCH_HOME",
                os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"),
            )
        )
    default_cache_path = os.path.join(torch_cache_home, default)

    cache_path = os.getenv(env_variable, default_cache_path)

    if not PathManager.exists(cache_path):
        try:
            PathManager.mkdirs(cache_path)
        except PermissionError:
            cache_path = os.path.join(get_mmf_root(), ".mmf_cache")
            PathManager.mkdirs(cache_path)

    return cache_path


def resolve_dir(env_variable, default="data"):
    default_dir = os.path.join(resolve_cache_dir(), default)
    dir_path = os.getenv(env_variable, default_dir)

    if not PathManager.exists(dir_path):
        PathManager.mkdirs(dir_path)

    return dir_path


class Configuration:
    def __init__(self, args=None, default_only=False):
        self.config = {}

        if not args:
            import argparse

            args = argparse.Namespace(opts=[])
            default_only = True

        self.args = args
        self._register_resolvers()

        self._default_config = self._build_default_config()

        if default_only:
            other_configs = {}
        else:
            other_configs = self._build_other_configs()

        self.config = OmegaConf.merge(self._default_config, other_configs)

        self.config = self._merge_with_dotlist(self.config, args.opts)
        self._update_specific(self.config)
        self.upgrade(self.config)
        # Resolve the config here itself after full creation so that spawned workers
        # don't face any issues
        self.config = OmegaConf.create(
            OmegaConf.to_container(self.config, resolve=True)
        )
        registry.register("config", self.config)

    def _build_default_config(self):
        self.default_config_path = get_default_config_path()
        default_config = load_yaml(self.default_config_path)
        return default_config

    def _build_other_configs(self):
        opts_config = self._build_opt_list(self.args.opts)
        user_config = self._build_user_config(opts_config)

        self._opts_config = opts_config
        self._user_config = user_config

        self.import_user_dir()

        model_config = self._build_model_config(opts_config)
        dataset_config = self._build_dataset_config(opts_config)
        args_overrides = self._build_demjson_config(self.args.config_override)
        other_configs = OmegaConf.merge(
            model_config, dataset_config, user_config, args_overrides
        )

        return other_configs

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

    def import_user_dir(self):
        # Try user_dir options in order of MMF configuration hierarchy
        # First try the default one, which can be set via environment as well
        user_dir = self._default_config.env.user_dir

        # Now, check user's config
        user_config_user_dir = self._user_config.get("env", {}).get("user_dir", None)

        if user_config_user_dir:
            user_dir = user_config_user_dir

        # Finally, check opts
        opts_user_dir = self._opts_config.get("env", {}).get("user_dir", None)
        if opts_user_dir:
            user_dir = opts_user_dir

        if user_dir:
            import_user_module(user_dir)

    def _build_model_config(self, config):
        model = config.model
        if model is None:
            raise KeyError("Required argument 'model' not passed")
        model_cls = registry.get_model_class(model)

        if model_cls is None:
            warning = f"No model named '{model}' has been registered"
            warnings.warn(warning)
            return OmegaConf.create()

        default_model_config_path = model_cls.config_path()

        if default_model_config_path is None:
            warning = "Model {}'s class has no default configuration provided".format(
                model
            )
            warnings.warn(warning)
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
                warning = f"No dataset named '{dataset}' has been registered"
                warnings.warn(warning)
                continue
            default_dataset_config_path = builder_cls.config_path()

            if default_dataset_config_path is None:
                warning = (
                    "Dataset {}'s builder class has no default configuration "
                    + f"provided"
                )
                warnings.warn(warning)
                continue

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
        OmegaConf.register_resolver("resolve_cache_dir", resolve_cache_dir)
        OmegaConf.register_resolver("resolve_dir", resolve_dir)

    def _merge_with_dotlist(self, config, opts):
        # TODO: To remove technical debt, a possible solution is to use
        # struct mode to update with dotlist OmegaConf node. Look into this
        # in next iteration
        if opts is None:
            opts = []

        if len(opts) == 0:
            return config

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
                    if (
                        not isinstance(
                            current_value,
                            (collections.abc.Mapping, collections.abc.Sequence),
                        )
                        or idx == len(splits) - 1
                    ):
                        logger.info(f"Overriding option {opt} to {value}")
                        current[stripped_field][array_index] = self._decode_value(value)
                    else:
                        # Otherwise move on down the chain
                        current = current_value
                else:
                    if idx == len(splits) - 1:
                        logger.info(f"Overriding option {opt} to {value}")
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

        logger.info("=====  Training Parameters    =====")
        logger.info(self._convert_node_to_json(self.config.training))

        logger.info("======  Dataset Attributes  ======")
        datasets = self.config.datasets.split(",")

        for dataset in datasets:
            if dataset in self.config.dataset_config:
                logger.info(f"======== {dataset} =======")
                dataset_config = self.config.dataset_config[dataset]
                logger.info(self._convert_node_to_json(dataset_config))
            else:
                logger.warning(f"No dataset named '{dataset}' in config. Skipping")

        logger.info("======  Optimizer Attributes  ======")
        logger.info(self._convert_node_to_json(self.config.optimizer))

        if self.config.model not in self.config.model_config:
            raise ValueError(f"{self.config.model} not present in model attributes")

        logger.info(f"======  Model ({self.config.model}) Attributes  ======")
        logger.info(
            self._convert_node_to_json(self.config.model_config[self.config.model])
        )

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def _update_specific(self, config):
        # tp = self.config.training

        # if args["seed"] is not None or tp['seed'] is not None:
        #     print(
        #         "You have chosen to seed the training. This will turn on CUDNN "
        #         "deterministic setting which can slow down your training "
        #         "considerably! You may see unexpected behavior when restarting "
        #         "from checkpoints."
        #     )

        # if args["seed"] == -1:
        #     self.config["training"]["seed"] = random.randint(1, 1000000)

        if config.learning_rate:
            if "optimizer" in config and "params" in config.optimizer:
                lr = config.learning_rate
                config.optimizer.params.lr = lr

        if not torch.cuda.is_available() and "cuda" in config.training.device:
            warnings.warn(
                "Device specified is 'cuda' but cuda is not present. "
                + "Switching to CPU version."
            )
            config.training.device = "cpu"

        return config

    def upgrade(self, config):
        mapping = {
            "training.resume_file": "checkpoint.resume_file",
            "training.resume": "checkpoint.resume",
            "training.resume_best": "checkpoint.resume_best",
            "training.load_pretrained": "checkpoint.resume_pretrained",
            "training.pretrained_state_mapping": "checkpoint.pretrained_state_mapping",
            "training.run_type": "run_type",
        }

        for old, new in mapping.items():
            value = OmegaConf.select(config, old)
            if value:
                OmegaConf.update(config, new, value)


# This is still here due to legacy reasons around
# older checkpoint loading from v0.3
class ConfigNode(collections.OrderedDict):
    pass
