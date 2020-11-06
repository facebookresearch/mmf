# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import importlib
import logging
import os
import sys
import warnings

import torch
from mmf.common.registry import registry
from mmf.utils.configuration import get_mmf_env, load_yaml
from mmf.utils.distributed import is_master, synchronize
from mmf.utils.download import download_pretrained_model
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_current_device, updir
from omegaconf import OmegaConf


try:
    import git
except ImportError:
    git = None

logger = logging.getLogger(__name__)
ALLOWED_CHECKPOINT_EXTS = [".ckpt", ".pth", ".pt"]


def _hack_imports():
    # NOTE: This can probably be made universal to support backwards
    # compatibility with name "pythia" if needed.
    sys.modules["pythia"] = importlib.import_module("mmf")
    sys.modules["pythia.utils.configuration"] = importlib.import_module(
        "mmf.utils.configuration"
    )


def _load_pretrained_checkpoint(checkpoint_path, *args, **kwargs):
    assert (
        os.path.splitext(checkpoint_path)[1] in ALLOWED_CHECKPOINT_EXTS
    ), f"Checkpoint must have extensions: {ALLOWED_CHECKPOINT_EXTS}"

    _hack_imports()

    with PathManager.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location=lambda storage, loc: storage)
    assert "config" in ckpt, (
        "No configs provided with pretrained model "
        " while checkpoint also doesn't have configuration."
    )
    config = ckpt.pop("config", None)
    model_config = config.get("model_config", config)

    ckpt = ckpt.get("model", ckpt)

    if "model_name" in kwargs:
        model_name = kwargs["model_name"]
    else:
        assert len(model_config.keys()) == 1, "Only one model type should be specified."
        model_name = list(model_config.keys())[0]

    model_config = model_config.get(model_name)
    return {"config": model_config, "checkpoint": ckpt, "full_config": config}


def _load_pretrained_model(model_name_or_path, *args, **kwargs):
    if PathManager.exists(model_name_or_path):
        download_path = model_name_or_path
        model_name = model_name_or_path
    else:
        download_path = download_pretrained_model(model_name_or_path, *args, **kwargs)
        model_name = model_name_or_path

    configs = glob.glob(os.path.join(download_path, "*.yaml"))
    assert len(configs) <= 1, (
        "Multiple yaml files with the pretrained model. "
        + "MMF doesn't know what to do."
    )

    ckpts = []
    allowed_ckpt_types = [f"*{ext}" for ext in ALLOWED_CHECKPOINT_EXTS]
    for ckpt_type in allowed_ckpt_types:
        ckpts.extend(glob.glob(os.path.join(download_path, ckpt_type)))

    assert (
        len(ckpts) == 1
    ), "None or multiple checkpoints files. MMF doesn't know what to do."

    _hack_imports()

    with PathManager.open(ckpts[0], "rb") as f:
        ckpt = torch.load(f, map_location=lambda storage, loc: storage)
    # If configs are not present, will ckpt provide the config?
    if len(configs) == 0:
        assert "config" in ckpt, (
            "No configs provided with pretrained model"
            " while checkpoint also doesn't have configuration."
        )
        config = ckpt["config"]
    else:
        config = load_yaml(configs[0])

    model_config = config.get("model_config", config)
    ckpt = ckpt.get("model", ckpt)
    # Also handle the case of model_name is path
    model_config = model_config.get(model_name.split(os.path.sep)[-1].split(".")[0])
    return {"config": model_config, "checkpoint": ckpt, "full_config": config}


def load_pretrained_model(model_name_or_path_or_checkpoint, *args, **kwargs):
    # If this is a file, then load this directly else download and load
    if PathManager.isfile(model_name_or_path_or_checkpoint):
        return _load_pretrained_checkpoint(
            model_name_or_path_or_checkpoint, args, kwargs
        )
    else:
        return _load_pretrained_model(model_name_or_path_or_checkpoint, args, kwargs)


def consolidate_optim_state_dict(optimizer):
    if hasattr(optimizer, "consolidate_state_dict"):
        optimizer.consolidate_state_dict(recipient_rank=0)


class Checkpoint:
    def __init__(self, trainer):
        """
        Generates a path for saving model which can also be used for resuming
        from a checkpoint.
        """
        self.trainer = trainer

        self.config = self.trainer.config
        self.save_dir = get_mmf_env(key="save_dir")
        self.model_name = self.config.model
        self.ckpt_foldername = self.save_dir
        self.device = get_current_device()
        self.ckpt_prefix = ""

        if hasattr(self.trainer.model, "get_ckpt_name"):
            self.ckpt_prefix = self.trainer.model.get_ckpt_name() + "_"

        self.pth_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + self.model_name + "_final.pth"
        )

        self.models_foldername = os.path.join(self.ckpt_foldername, "models")
        if not PathManager.exists(self.models_foldername):
            PathManager.mkdirs(self.models_foldername)

        self.save_config()

        self.repo_path = updir(os.path.abspath(__file__), n=3)
        self.git_repo = None
        if git and self.config.checkpoint.save_git_details:
            try:
                self.git_repo = git.Repo(self.repo_path)
            except git.exc.InvalidGitRepositoryError:
                # Not a git repo, don't do anything
                pass

        self.max_to_keep = self.config.checkpoint.max_to_keep
        self.saved_iterations = []

    def save_config(self):
        cfg_file = os.path.join(self.ckpt_foldername, "config.yaml")
        with PathManager.open(cfg_file, "w") as f:
            f.write(self.config.pretty(resolve=True))

    def load_state_dict(self):
        ckpt_config = self.config.checkpoint

        suffix = "best.ckpt" if ckpt_config.resume_best else "current.ckpt"
        reverse_suffix = "best.ckpt" if not ckpt_config.resume_best else "current.ckpt"
        ckpt_filepath = os.path.join(self.ckpt_foldername, self.ckpt_prefix + suffix)

        # In case of interrupts and resume, ckpt_config.resume_file would be there
        # But, if the checkpoints are already created in the save dir
        # and resume is true signifying the interrupt resume, we should skip
        # loading the resume file.
        if (
            ckpt_config.resume_file is not None or ckpt_config.resume_zoo is not None
        ) and (not ckpt_config.resume or not PathManager.exists(ckpt_filepath)):
            if ckpt_config.resume_file and PathManager.exists(ckpt_config.resume_file):
                self._load(
                    ckpt_config.resume_file,
                    load_pretrained=ckpt_config.resume_pretrained,
                )
                return
            # resume_file doesn't exist, try from zoo now
            elif ckpt_config.resume_zoo is not None:
                self._load(
                    ckpt_config.resume_zoo,
                    load_zoo=True,
                    load_pretrained=ckpt_config.resume_pretrained,
                )
                return
            else:
                raise RuntimeError(f"{ckpt_config.resume_file} doesn't exist")

        if ckpt_config.resume:
            if PathManager.exists(ckpt_filepath):
                self._load(ckpt_filepath)
            else:
                warnings.warn(
                    "Tried to resume but checkpoint filepath {} "
                    "is not present. Trying {}, otherwise skipping.".format(
                        ckpt_filepath, reverse_suffix
                    )
                )
                ckpt_filepath = ckpt_filepath.replace(suffix, reverse_suffix)
                if PathManager.exists(ckpt_filepath):
                    self._load(ckpt_filepath)

    def _load(self, file, force=False, load_zoo=False, load_pretrained=False):
        ckpt_config = self.config.checkpoint
        logger.info("Loading checkpoint")
        if load_zoo:
            ckpt, should_continue = self._load_from_zoo(file)
            if not should_continue:
                return
        else:
            ckpt = self._torch_load(file)

        if "model" not in ckpt:
            ckpt = {"model": ckpt}

        pretrained_state_mapping = ckpt_config.pretrained_state_mapping

        if not load_pretrained or force is True:
            pretrained_state_mapping = {}

        state_dict = self.upgrade_state_dict(ckpt["model"])

        if len(pretrained_state_mapping.items()) == 0:
            incompatible_keys = self.trainer.model.load_state_dict(
                state_dict, strict=False
            )
            if len(incompatible_keys.missing_keys) != 0:
                logger.warning(
                    f"Missing keys {incompatible_keys.missing_keys} in the"
                    + " checkpoint.\n"
                    + "If this is not your checkpoint, please open up an "
                    + "issue on MMF GitHub. \n"
                    + f"Unexpected keys if any: {incompatible_keys.unexpected_keys}"
                )

            if len(incompatible_keys.unexpected_keys) != 0:
                logger.warning(
                    "Unexpected keys in state dict: "
                    + f"{incompatible_keys.unexpected_keys} \n"
                    + "This is usually not a problem with pretrained models, but "
                    + "if this is your own model, please double check. \n"
                    + "If you think this is an issue, please open up a "
                    + "bug at MMF GitHub."
                )

            reset_optimizer = ckpt_config.reset.optimizer or ckpt_config.reset.all
            if not reset_optimizer:
                self._load_optimizer(ckpt)

            reset_counts = ckpt_config.reset.all or ckpt_config.reset.counts
            if not reset_counts:
                self.trainer.early_stop_callback.early_stopping.init_from_checkpoint(
                    ckpt
                )
                self._load_counts_and_lr_scheduler(ckpt)

            reset_scaler = ckpt_config.reset.all or ckpt_config.reset.fp16_scaler
            if not reset_scaler:
                self._load_fp16_scaler(ckpt)
        else:
            self._load_pretrained(state_dict)

        logger.info("Checkpoint loaded.")
        logger.info(f"Current num updates: {self.trainer.num_updates}")
        logger.info(f"Current iteration: {self.trainer.current_iteration}")
        logger.info(f"Current epoch: {self.trainer.current_epoch}")

    def _load_optimizer(self, ckpt):
        if "optimizer" in ckpt:
            try:
                self.trainer.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                logger.info(
                    "Optimizer failed to load. Try with "
                    + "checkpoint.reset.optimizer=True"
                )
                raise
        else:
            warnings.warn(
                "'optimizer' key is not present in the "
                "checkpoint asked to be loaded. Skipping."
            )

    def _load_counts_and_lr_scheduler(self, ckpt):
        ckpt_config = self.trainer.config.checkpoint
        if "best_update" in ckpt:
            if ckpt_config.resume_best:
                self.trainer.num_updates = ckpt.get(
                    "best_update", self.trainer.num_updates
                )
                self.trainer.current_iteration = ckpt.get(
                    "best_iteration", self.trainer.current_iteration
                )
            else:
                self.trainer.num_updates = ckpt.get(
                    "num_updates", self.trainer.num_updates
                )
                self.trainer.current_iteration = ckpt.get(
                    "current_iteration", self.trainer.current_iteration
                )

            self.trainer.current_epoch = ckpt.get(
                "current_epoch", self.trainer.current_epoch
            )
        elif "best_iteration" in ckpt:
            # Preserve old behavior for old checkpoints where we always
            # load best iteration
            if ckpt_config.resume_best and "current_iteration" in ckpt:
                self.trainer.current_iteration = ckpt["current_iteration"]
            else:
                self.trainer.current_iteration = ckpt.get(
                    "best_iteration", self.trainer.current_iteration
                )

            self.trainer.num_updates = self.trainer.current_iteration

        lr_scheduler = self.trainer.lr_scheduler_callback._scheduler
        if lr_scheduler is not None:
            if "lr_scheduler" in ckpt:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            else:
                warnings.warn(
                    "'lr_scheduler' key is not present in the "
                    "checkpoint asked to be loaded. Setting lr_scheduler's "
                    "last_epoch to current_iteration."
                )
                lr_scheduler.last_epoch = self.trainer.current_iteration

        registry.register("current_iteration", self.trainer.current_iteration)
        registry.register("num_updates", self.trainer.num_updates)

        self.trainer.current_epoch = ckpt.get("best_epoch", self.trainer.current_epoch)
        registry.register("current_epoch", self.trainer.current_epoch)

    def _load_fp16_scaler(self, ckpt):
        scaler = getattr(self.trainer, "scaler", None)
        scaler_dict = ckpt.get("fp16_scaler", None)
        if scaler is not None and scaler_dict is not None:
            scaler.load_state_dict(scaler_dict)

    def _load_pretrained(self, ckpt):
        model = self.trainer.model
        own_state = model.state_dict()
        mapping = self.trainer.config.checkpoint.pretrained_state_mapping
        for key, value in mapping.items():
            key += "."
            value += "."
            for attr in ckpt:
                if hasattr(model, "format_state_key"):
                    formatted_attr = model.format_state_key(attr)
                else:
                    formatted_attr = attr

                for own_attr in own_state:
                    if (
                        key in own_attr
                        and value in formatted_attr
                        and own_attr.replace(key, "")
                        == formatted_attr.replace(value, "")
                    ):
                        logger.info("Copying " + own_attr + " from " + attr)
                        own_state[own_attr].copy_(ckpt[attr])
        logger.info("Pretrained model loaded")

    def upgrade_state_dict(self, state_dict):
        data_parallel = registry.get("data_parallel") or registry.get("distributed")
        data_parallel = data_parallel or isinstance(
            self.trainer.model,
            (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
        )
        if data_parallel:
            model = self.trainer.model.module
        else:
            model = self.trainer.model

        new_dict = {}
        for attr in state_dict:
            new_attr = model.format_state_key(attr)
            if not data_parallel and attr.startswith("module."):
                # In case the ckpt was actually a data parallel model
                # replace first module. from dataparallel with empty string
                new_attr = new_attr.replace("module.", "", 1)
            elif data_parallel and not attr.startswith("module."):
                new_attr = "module." + new_attr

            # Log if key has changed but not when the difference
            # is only due to data parallel's `module`
            if new_attr != attr and ("module." + new_attr != attr):
                logger.info(f"Will load key {new_attr} from {attr}")
            new_dict[new_attr] = state_dict[attr]
        return new_dict

    def _load_from_zoo(self, file):
        ckpt_config = self.trainer.config.checkpoint
        zoo_ckpt = load_pretrained_model(file)

        # If zoo_config_override, load the model directly using `from_pretrained`
        if ckpt_config.zoo_config_override:
            model_cls = registry.get_model_class(self.trainer.config.model)
            self.trainer.model = model_cls.from_pretrained(ckpt_config.resume_zoo)
            self.trainer.config.model_config = zoo_ckpt["full_config"].model_config
            return None, False
        else:
            return self.upgrade_state_dict(zoo_ckpt["checkpoint"]), True

    def _torch_load(self, file):
        # Backwards compatibility to Pythia
        _hack_imports()

        with PathManager.open(file, "rb") as f:
            if "cuda" in str(self.device):
                return torch.load(f, map_location=self.device)
            else:
                return torch.load(f, map_location=lambda storage, loc: storage)

    def _get_vcs_fields(self):
        """Returns a dict with git fields of the current repository

           To reproduce an experiment directly from a checkpoint

           1) Export `config` key as a yaml
           2) Clone repository and checkout at given commit on given branch
           3) Any local change (diff) while running the experiment is stored
              in the value with key `git/diff`, output the diff to a `path.diff`
              file and apply the patch to the current state by simply

                           `patch -p0 < path.diff`
        """

        return {
            "git/branch": self.git_repo.active_branch.name,
            "git/commit_hash": self.git_repo.head.commit.name_rev,
            "git/commit_author": self.git_repo.head.commit.author.name,
            "git/commit_message": self.git_repo.head.commit.message,
            "git/diff": self.git_repo.git.diff("--no-prefix"),
        }

    def save(self, update, iteration=None, update_best=False):
        # Only save in main process
        if not is_master():
            return

        if not iteration:
            iteration = update

        ckpt_filepath = os.path.join(self.models_foldername, "model_%d.ckpt" % update)
        best_ckpt_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + "best.ckpt"
        )
        current_ckpt_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + "current.ckpt"
        )

        best_iteration = (
            self.trainer.early_stop_callback.early_stopping.best_monitored_iteration
        )
        best_update = (
            self.trainer.early_stop_callback.early_stopping.best_monitored_update
        )
        best_metric = (
            self.trainer.early_stop_callback.early_stopping.best_monitored_value
        )
        model = self.trainer.model
        data_parallel = registry.get("data_parallel") or registry.get("distributed")
        fp16_scaler = getattr(self.trainer, "scaler", None)
        fp16_scaler_dict = None

        if fp16_scaler is not None:
            fp16_scaler_dict = fp16_scaler.state_dict()

        if data_parallel is True:
            model = model.module

        ckpt = {
            "model": model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "best_iteration": best_iteration,
            "current_iteration": iteration,
            "current_epoch": self.trainer.current_epoch,
            "num_updates": update,
            "best_update": best_update,
            "best_metric_value": best_metric,
            "fp16_scaler": fp16_scaler_dict,
            # Convert to container to avoid any dependencies
            "config": OmegaConf.to_container(self.config, resolve=True),
        }

        lr_scheduler = self.trainer.lr_scheduler_callback._scheduler
        if lr_scheduler is not None:
            ckpt["lr_scheduler"] = lr_scheduler.state_dict()

        if self.git_repo:
            git_metadata_dict = self._get_vcs_fields()
            ckpt.update(git_metadata_dict)

        with PathManager.open(ckpt_filepath, "wb") as f:
            torch.save(ckpt, f)

        if update_best:
            with PathManager.open(best_ckpt_filepath, "wb") as f:
                torch.save(ckpt, f)

        # Save current always
        with PathManager.open(current_ckpt_filepath, "wb") as f:
            torch.save(ckpt, f)

        # Remove old checkpoints if max_to_keep is set
        if self.max_to_keep > 0:
            if len(self.saved_iterations) == self.max_to_keep:
                self.remove(self.saved_iterations.pop(0))
            self.saved_iterations.append(update)

    def remove(self, update):
        ckpt_filepath = os.path.join(self.models_foldername, "model_%d.ckpt" % update)
        if PathManager.isfile(ckpt_filepath):
            PathManager.rm(ckpt_filepath)

    def restore(self):
        synchronize()
        logger.info("Restoring checkpoint")
        best_path = os.path.join(self.ckpt_foldername, self.ckpt_prefix + "best.ckpt")

        if PathManager.exists(best_path):
            self._load(best_path, force=True)

    def finalize(self):
        if is_master():
            with PathManager.open(self.pth_filepath, "wb") as f:
                torch.save(self.trainer.model.state_dict(), f)
