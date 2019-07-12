# Copyright (c) Facebook, Inc. and its affiliates.
import os
import warnings

import git
import torch
import yaml

from pythia.common.registry import registry
from pythia.utils.distributed_utils import is_main_process, synchronize
from pythia.utils.general import (ckpt_name_from_core_args,
                                  foldername_from_config_override, updir)


class Checkpoint:
    def __init__(self, trainer):
        """
        Generates a path for saving model which can also be used for resuming
        from a checkpoint.
        """
        self.trainer = trainer

        self.config = self.trainer.config
        self.save_dir = self.config.training_parameters.save_dir
        self.model_name = self.config.model

        self.ckpt_foldername = ckpt_name_from_core_args(self.config)
        self.ckpt_foldername += foldername_from_config_override(self.trainer.args)

        self.device = registry.get("current_device")

        self.ckpt_prefix = ""

        if hasattr(self.trainer.model, "get_ckpt_name"):
            self.ckpt_prefix = self.trainer.model.get_ckpt_name() + "_"

        self.config["log_foldername"] = self.ckpt_foldername
        self.ckpt_foldername = os.path.join(self.save_dir, self.ckpt_foldername)
        self.pth_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + self.model_name + "_final.pth"
        )

        self.models_foldername = os.path.join(self.ckpt_foldername, "models")
        if not os.path.exists(self.models_foldername):
            os.makedirs(self.models_foldername)

        self.save_config()
        self.repo_path = updir(os.path.abspath(__file__), n=3)
        self.repo = git.Repo(self.repo_path)

    def save_config(self):
        cfg_file = os.path.join(self.ckpt_foldername, "config.yaml")
        with open(cfg_file, "w") as f:
            # Pop out config_override if present to remove clutter in
            # saved configuration yaml file
            self.config.pop("config_override", None)
            f.write(str(self.config))

    def load_state_dict(self):
        tp = self.config.training_parameters
        if tp.resume_file is not None:
            if os.path.exists(tp.resume_file):
                self._load(tp.resume_file)
                return
            else:
                raise RuntimeError("{} doesn't exist".format(tp.resume_file))

        ckpt_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + "best.ckpt"
        )

        if tp.resume is True:
            if os.path.exists(ckpt_filepath):
                self._load(ckpt_filepath)
            else:
                warnings.warn(
                    "Tried to resume but checkpoint filepath {} "
                    "is not present. Skipping.".format(ckpt_filepath)
                )

    def _load(self, file):
        self.trainer.writer.write("Loading checkpoint")
        ckpt = self._torch_load(file)

        data_parallel = registry.get("data_parallel")

        if "model" in ckpt:
            ckpt_model = ckpt["model"]
        else:
            ckpt_model = ckpt
            ckpt = {"model": ckpt}

        pretrained_mapping = self.config.training_parameters.pretrained_mapping

        if not self.config.training_parameters.load_pretrained:
            pretrained_mapping = {}

        new_dict = {}

        # TODO: Move to separate function
        for attr in ckpt_model:
            if "fa_history" in attr:
                new_dict[attr.replace("fa_history", "fa_context")] = ckpt_model[attr]
            elif data_parallel is False and attr.startswith("module."):
                # In case the ckpt was actually a data parallel model
                # replace first module. from dataparallel with empty string
                new_dict[attr.replace("module.", "", 1)] = ckpt_model[attr]
            else:
                new_dict[attr] = ckpt_model[attr]

        if len(pretrained_mapping.items()) == 0:
            final_dict = new_dict

            self.trainer.model.load_state_dict(final_dict)

            if "optimizer" in ckpt:
                self.trainer.optimizer.load_state_dict(ckpt["optimizer"])
            else:
                warnings.warn(
                    "'optimizer' key is not present in the "
                    "checkpoint asked to be loaded. Skipping."
                )

            self.trainer.early_stopping.init_from_checkpoint(ckpt)

            self.trainer.writer.write("Checkpoint loaded")

            if "best_iteration" in ckpt:
                self.trainer.current_iteration = ckpt["best_iteration"]
                registry.register("current_iteration", self.trainer.current_iteration)

            if "best_epoch" in ckpt:
                self.trainer.current_epoch = ckpt["best_epoch"]
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
                        if (
                            key in attr
                            and value in own_attr
                            and attr.replace(key, "") == own_attr.replace(value, "")
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
            return torch.load(file)
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
            "git/branch": self.repo.active_branch.name,
            "git/commit_hash": self.repo.head.commit.name_rev,
            "git/commit_author": self.repo.head.commit.author.name,
            "git/commit_message": self.repo.head.commit.message,
            "git/diff": self.repo.git.diff("--no-prefix"),
        }

    def save(self, iteration, update_best=False):
        # Only save in main process
        if not is_main_process():
            return

        ckpt_filepath = os.path.join(
            self.models_foldername, "model_%d.ckpt" % iteration
        )
        best_ckpt_filepath = os.path.join(
            self.ckpt_foldername, self.ckpt_prefix + "best.ckpt"
        )

        best_iteration = self.trainer.early_stopping.best_monitored_iteration
        best_metric = self.trainer.early_stopping.best_monitored_value

        ckpt = {
            "model": self.trainer.model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "best_iteration": best_iteration,
            "best_metric_value": best_metric,
            "config": self.config,
        }

        git_metadata_dict = self._get_vcs_fields()
        ckpt.update(git_metadata_dict)

        torch.save(ckpt, ckpt_filepath)

        if update_best:
            torch.save(ckpt, best_ckpt_filepath)

    def restore(self):
        self.trainer.writer.write("Restoring checkpoint")
        best_path = os.path.join(self.ckpt_foldername, self.ckpt_prefix + "best.ckpt")

        if os.path.exists(best_path):
            ckpt = self._torch_load(best_path)
            self.trainer.model.load_state_dict(ckpt["model"])
            self.trainer.optimizer.load_state_dict(ckpt["optimizer"])

    def finalize(self):
        torch.save(self.trainer.model.state_dict(), self.pth_filepath)
