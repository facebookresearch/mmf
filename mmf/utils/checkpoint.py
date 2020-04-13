# Copyright (c) Facebook, Inc. and its affiliates.

import os
import warnings

import git
import torch
from omegaconf import OmegaConf

from mmf.common.registry import registry
from mmf.utils.distributed import is_master, synchronize
from mmf.utils.general import updir


class Checkpoint:
    def __init__(self, trainer):
        """
        Generates a path for saving model which can also be used for resuming
        from a checkpoint.
        """
        self.trainer = trainer

        self.config = self.trainer.config
        self.save_dir = self.config.training.save_dir
        self.model_name = self.config.model

        self.ckpt_foldername = self.save_dir

        self.device = registry.get("current_device")

        self.ckpt_prefix = ""

        if hasattr(self.trainer.model, "get_ckpt_name"):
            self.ckpt_prefix = self.trainer.model.get_ckpt_name() + "_"

        self.pth_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + self.model_name + "_final.pth"
        )

        self.models_foldername = os.path.join(self.ckpt_foldername, "models")
        if not os.path.exists(self.models_foldername):
            os.makedirs(self.models_foldername, exist_ok=True)

        self.save_config()
        self.repo_path = updir(os.path.abspath(__file__), n=3)
        self.git_repo = None
        try:
            self.git_repo = git.Repo(self.repo_path)
        except git.exc.InvalidGitRepositoryError:
            self.git_repo = None

    def save_config(self):
        cfg_file = os.path.join(self.ckpt_foldername, "config.yaml")
        with open(cfg_file, "w") as f:
            # Pop out config_override if present to remove clutter in
            # saved configuration yaml file
            self.config.pop("config_override", None)
            f.write(self.config.pretty(resolve=True))

    def load_state_dict(self):
        tp = self.config.training

        suffix = "best.ckpt" if tp.resume_best else "current.ckpt"
        reverse_suffix = "best.ckpt" if not tp.resume_best else "current.ckpt"
        ckpt_filepath = os.path.join(self.ckpt_foldername, self.ckpt_prefix + suffix)

        # In case of interrupts and resume, tp.resume_file would be there
        # But, if the checkpoints are already created in the save dir
        # and resume is true signifying the interrupt resume, we should skip
        # loading the resume file.
        if tp.resume_file is not None and (
            tp.resume is False or not os.path.exists(ckpt_filepath)
        ):
            if os.path.exists(tp.resume_file):
                self._load(tp.resume_file, load_pretrained=tp.load_pretrained)
                return
            else:
                raise RuntimeError("{} doesn't exist".format(tp.resume_file))

        if tp.resume is True:
            if os.path.exists(ckpt_filepath):
                self._load(ckpt_filepath)
            else:
                warnings.warn(
                    "Tried to resume but checkpoint filepath {} "
                    "is not present. Trying {}, otherwise skipping.".format(
                        ckpt_filepath, reverse_suffix
                    )
                )
                ckpt_filepath = ckpt_filepath.replace(suffix, reverse_suffix)
                if os.path.exists(ckpt_filepath):
                    self._load(ckpt_filepath)

    def _load(self, file, force=False, load_pretrained=False):
        tp = self.config.training
        self.trainer.writer.write("Loading checkpoint")

        ckpt = self._torch_load(file)

        data_parallel = registry.get("data_parallel") or registry.get("distributed")

        if "model" in ckpt:
            ckpt_model = ckpt["model"]
        else:
            ckpt_model = ckpt
            ckpt = {"model": ckpt}

        pretrained_mapping = tp.pretrained_mapping

        if load_pretrained is False or force is True:
            pretrained_mapping = {}

        new_dict = {}

        # TODO: Move to separate function
        for attr in ckpt_model:
            new_attr = attr
            if "fa_history" in attr:
                new_attr = new_attr.replace("fa_history", "fa_context")

            if data_parallel is False and attr.startswith("module."):
                # In case the ckpt was actually a data parallel model
                # replace first module. from dataparallel with empty string
                new_dict[new_attr.replace("module.", "", 1)] = ckpt_model[attr]
            elif data_parallel is not False and not attr.startswith("module."):
                new_dict["module." + new_attr] = ckpt_model[attr]
            else:
                new_dict[new_attr] = ckpt_model[attr]

        if len(pretrained_mapping.items()) == 0:
            final_dict = new_dict
            self.trainer.model.load_state_dict(final_dict, strict=False)

            if "optimizer" in ckpt:
                self.trainer.optimizer.load_state_dict(ckpt["optimizer"])
            else:
                warnings.warn(
                    "'optimizer' key is not present in the "
                    "checkpoint asked to be loaded. Skipping."
                )

            self.trainer.early_stopping.init_from_checkpoint(ckpt)

            self.trainer.writer.write("Checkpoint loaded")

            if "best_update" in ckpt:
                if tp.resume_best:
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
                # Preserve old behavior for old checkpoints where we always load best iteration
                if tp.resume_best and "current_iteration" in ckpt:
                    self.trainer.current_iteration = ckpt["current_iteration"]
                else:
                    self.trainer.current_iteration = ckpt.get(
                        "best_iteration", self.trainer.current_iteration
                    )

                self.trainer.num_updates = self.trainer.current_iteration

            registry.register("current_iteration", self.trainer.current_iteration)
            registry.register("num_updates", self.trainer.num_updates)

            self.trainer.current_epoch = ckpt.get(
                "best_epoch", self.trainer.current_epoch
            )
            registry.register("current_epoch", self.trainer.current_epoch)
        else:
            final_dict = {}
            model = self.trainer.model
            own_state = model.state_dict()

            for key, value in pretrained_mapping.items():
                key += "."
                value += "."
                for attr in new_dict:
                    for own_attr in own_state:
                        formatted_attr = model.format_state_key(attr)
                        if (
                            key in formatted_attr
                            and value in own_attr
                            and formatted_attr.replace(key, "")
                            == own_attr.replace(value, "")
                        ):
                            self.trainer.writer.write(
                                "Copying " + attr + " " + own_attr
                            )
                            own_state[own_attr].copy_(new_dict[attr])
            self.trainer.writer.write("Pretrained model loaded")

    def _load_state_dict_mapping(self, ckpt_model):
        model = self.trainer.model
        attr_mapping = {
            "image_feature_encoders": "img_feat_encoders",
            "image_feature_embeddings_list": "img_embeddings_list",
            "image_text_multi_modal_combine_layer": "multi_modal_combine_layer",
            "text_embeddings": "text_embeddings",
            "classifier": "classifier",
        }

        data_parallel = registry.get("data_parallel")

        if not data_parallel:
            for key in attr_mapping:
                attr_mapping[key.replace("module.", "")] = attr_mapping[key]
                attr_mapping.pop(key)

        for key in attr_mapping:
            getattr(model, key).load_state_dict(ckpt_model[attr_mapping[key]])

    def _torch_load(self, file):
        if "cuda" in str(self.device):
            return torch.load(file, map_location=self.device)
        else:
            return torch.load(file, map_location=lambda storage, loc: storage)

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

    def save(self, update, iteration, update_best=False):
        # Only save in main process
        if not is_master():
            return

        ckpt_filepath = os.path.join(self.models_foldername, "model_%d.ckpt" % update)
        best_ckpt_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + "best.ckpt"
        )
        current_ckpt_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + "current.ckpt"
        )

        best_iteration = self.trainer.early_stopping.best_monitored_iteration
        best_update = self.trainer.early_stopping.best_monitored_update
        best_metric = self.trainer.early_stopping.best_monitored_value
        model = self.trainer.model
        data_parallel = registry.get("data_parallel") or registry.get("distributed")

        if data_parallel is True:
            model = model.module

        ckpt = {
            "model": model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "best_iteration": best_iteration,
            "current_iteration": registry.get("current_iteration"),
            "current_epoch": self.trainer.current_epoch,
            "num_updates": registry.get("num_updates"),
            "best_update": best_update,
            "best_metric_value": best_metric,
            # Convert to container to avoid any dependencies
            "config": OmegaConf.to_container(self.config, resolve=True),
        }

        if self.git_repo:
            git_metadata_dict = self._get_vcs_fields()
            ckpt.update(git_metadata_dict)

        torch.save(ckpt, ckpt_filepath)

        if update_best:
            torch.save(ckpt, best_ckpt_filepath)

        # Save current always
        torch.save(ckpt, current_ckpt_filepath)

    def restore(self):
        synchronize()
        self.trainer.writer.write("Restoring checkpoint")
        best_path = os.path.join(self.ckpt_foldername, self.ckpt_prefix + "best.ckpt")

        if os.path.exists(best_path):
            self._load(best_path, force=True)

    def finalize(self):
        torch.save(self.trainer.model.state_dict(), self.pth_filepath)
