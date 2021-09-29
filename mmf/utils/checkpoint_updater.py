# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Any, Dict, Tuple

import torch
from mmf.common.registry import registry


logger = logging.getLogger(__name__)


def is_pl_model_checkpoint(checkpoint):
    return "state_dict" in checkpoint


def is_pl_trainer_checkpoint(checkpoint):
    return "pytorch-lightning_version" in checkpoint


def is_model_only_checkpoint(checkpoint):
    if is_pl_trainer_checkpoint(checkpoint):
        return "state_dict" not in checkpoint
    else:
        return "model" not in checkpoint


def _format_state_key(model: torch.nn.Module, attr: str):
    if hasattr(model, "format_state_key"):
        formatted_attr = model.format_state_key(attr)
    else:
        formatted_attr = attr

    return formatted_attr


def _should_skip_if_mismatch(
    shape1: Tuple[str, torch.Size],
    shape2: Tuple[str, torch.Size],
    config: Dict[str, Any],
) -> None:
    if shape1[1] != shape2[1]:
        message = f"""
            Modules {shape1[0]} and {shape2[0]} don't have the same shape:
            own_attr has shape {shape1[1]} while
            attr has shape {shape2[1]}. This can fail down the line.
            """
        if config.checkpoint.get("bypass_shape_mismatch", False):
            message += "bypass_shape_mismatch in config.checkpoint "
            message += "is set to be True, -- so skipping copy"
            logger.warning(message)

            return True
        else:
            logger.warning(message)

    # In case of either mismatch or match both, MMF will try
    # to copy the attribute.
    return False


def get_pretrained_state_mapping_checkpoint(
    checkpoint: Dict[str, Any], model: torch.nn.Module, config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    This function gets the checkpoint keys that exists in pretrained state mapping
    that also exist in model's state, and returns a dictionary with the value from the
    `checkpoint` dict.
    """
    mapping = config.checkpoint.pretrained_state_mapping
    own_state = model.state_dict()
    tmp_checkpoint = dict(checkpoint)

    ckpt_update_dict = dict()
    for key, value in mapping.items():
        key += "."
        value += "."
        for attr in tmp_checkpoint:
            formatted_attr = _format_state_key(model, attr)
            for own_attr in own_state:
                if (
                    key in own_attr
                    and value in formatted_attr
                    and own_attr.replace(key, "") == formatted_attr.replace(value, "")
                ):
                    if _should_skip_if_mismatch(
                        (own_attr, own_state[own_attr].shape),
                        (attr, checkpoint[attr].shape),
                        config,
                    ):
                        continue
                    ckpt_update_dict[own_attr] = attr
    return ckpt_update_dict


def remove_keys_inplace(ckpt: Dict[str, Any], keys_to_remove):
    tmp_keys = dict(ckpt)
    for key in tmp_keys:
        if key in keys_to_remove:
            ckpt.pop(key)


class MMFToPLCheckpointUpdater:
    def __init__(self):
        pass

    def update_checkpoint(
        self, checkpoint: Dict[str, Any], model: torch.nn.Module
    ) -> None:
        r"""
        This function should only be called on lightning. It handles checkpoint
        update that is being called by LightningModule's `on_load_checkpoint`,
        which should update the checkpoint to the format desired. The logic
        contains two parts, when checkpoint is a model only checkpoint and
        when checkpoint is a trainer checkpoint. This function applies the checkpoint
        update in place.

        If the checkpoint is a model only checkpoint:
            1. If it is an mmf checkpoint, convert to lightning format
                putting it inside a "state_dict" key
            2. Apply the model's format state key to give the model a chance to update
            3. If config.checkpoint.pretrained_state_mapping is True, apply
                the mapping speicified in the config, and remove the keys that exist
                in the checkpoint that do not exist in the mapping.
        The updated checkpoint should be of the format: {"state_dict": ckpts}, where
        ckpts should be the model state_dict.

        If the checkpoint is a trainer only checkpoint:
            1. do the above steps for model checkpoint update
            2. do the checkpoint trainer state update from mmf to lightning
        The updated checkpoint should be of the format: {
            `epoch`: x,
            `global_step`: x,
            `pytorch-lightning_version`: x,
            `state_dict`: x,
            `callbacks`: x,
            `optimizer_states`: [x],
            `lr_schedulers`: [x],
        }
        """

        if is_model_only_checkpoint(checkpoint):
            self._update_model_checkpoint(checkpoint=checkpoint, model=model)
            return

        # this assumes the checkpoint is trainer only
        if not is_pl_trainer_checkpoint(checkpoint):
            self._update_trainer_checkpoint_from_mmf(checkpoint=checkpoint, model=model)

    def _update_trainer_checkpoint_from_mmf(
        self, checkpoint: Dict[str, Any], model: Any
    ) -> None:
        """updates checkpoint from the mmf format to lightning format.
        mmf checkpoint is with keys:
        `model`, `optimizer`, `best_iteration`, `current_iteration`, `current_epoch`, ,
        `num_updates`, `best_update`, `best_metric_value`, `fp16_scaler`, `config`, ,
        `lr_scheduler`, `git/branch`, `git/commit_hash`, `git/commit_author`,
        `git/commit_message`, `git/diff`
        """
        remove_keys_inplace(
            checkpoint,
            {
                "best_iteration",
                "current_iteration",
                "best_update",
                "best_metric_value",
                "fp16_scaler",
                "config",
                "git/branch",
                "git/commit_hash",
                "git/commit_author",
                "git/commit_message",
                "git/diff",
            },
        )

        # update model
        if "model" in checkpoint:
            model_checkpoint = checkpoint.pop("model")
            checkpoint["state_dict"] = model_checkpoint
            self._update_model_format_state_keys(checkpoint["state_dict"], model=model)
            config = registry.get("config")
            if config.checkpoint.get("resume_pretrained", False):
                self._update_pretrained_state_mapping(
                    checkpoint=checkpoint["state_dict"], model=model, config=config
                )
        # update trainer progress
        if "optimizer" in checkpoint:
            optimizer = checkpoint.pop("optimizer")
            checkpoint["optimizer_states"] = [optimizer]
        if "lr_scheduler" in checkpoint:
            lr_scheduler = checkpoint.pop("lr_scheduler")
            checkpoint["lr_schedulers"] = [lr_scheduler]
        else:
            # we need to set this if it is not specified bc lightning expects
            # lr_schedulers to be present to resume checkpoint while in mmf, it is
            # not guranteed that lr_schedulers are used and saved in the checkpoint.
            checkpoint["lr_schedulers"] = []
        if "num_updates" in checkpoint:
            global_step = checkpoint.pop("num_updates")
            checkpoint["global_step"] = global_step
        if "current_epoch" in checkpoint:
            epoch = checkpoint.pop("current_epoch")
            checkpoint["epoch"] = epoch

    def _update_model_checkpoint(
        self, checkpoint: Dict[str, Any], model: torch.nn.Module
    ) -> None:
        """
        This function assumes the checkpoint is just the model and does not include
        training params.
        """
        if not is_pl_model_checkpoint(checkpoint):
            self._update_model_checkpoint_from_mmf(checkpoint)

        # this assumes that model_checkpoint here is the lightning format
        self._update_model_format_state_keys(checkpoint["state_dict"], model=model)

        config = registry.get("config")
        if config.checkpoint.get("resume_pretrained", False):
            self._update_pretrained_state_mapping(
                checkpoint=checkpoint["state_dict"], model=model, config=config
            )

    def _update_pretrained_state_mapping(
        self, checkpoint: Dict[str, Any], model: torch.nn.Module, config: Dict[str, Any]
    ) -> None:
        """
        This function removes all checkpoint keys that do not exist in
        the `pretrained_state_mapping`
        """
        ckpt_update_dict = get_pretrained_state_mapping_checkpoint(
            checkpoint=checkpoint, model=model, config=config
        )
        accepted_keys = set()
        for own_attr, attr in ckpt_update_dict.items():
            assert own_attr == attr, (
                "Since `_update_model_format_state_keys` was run ",
                "before, this has to be held true",
            )
            logger.info("Copying " + own_attr + " from " + attr)
            accepted_keys.add(attr)

        # keep only the checkpoint keys that exist in the `pretrained_state_mapping`
        tmp_checkpoint = dict(checkpoint)
        for key in tmp_checkpoint:
            if key not in accepted_keys:
                checkpoint.pop(key)

    def _update_model_format_state_keys(
        self, checkpoint: Dict[str, Any], model: torch.nn.Module
    ) -> None:
        """
        Function to rewrite the checkpoint in place to give the model a chance
        to update state_dict keys. This assumes that checkpoint is the
        model's state_dict.
        """
        tmp_state_dict = dict(checkpoint)
        for attr in tmp_state_dict:
            new_attr = _format_state_key(model, attr)
            if attr != new_attr:
                logger.info(f"checkpoint: rewriting {attr} into {new_attr}")
                value = checkpoint.pop(attr)
                checkpoint[new_attr] = value

    def _update_model_checkpoint_from_mmf(self, checkpoint: Dict[str, Any]) -> None:
        tmp_checkpoint = dict(checkpoint)
        checkpoint.clear()
        checkpoint["state_dict"] = tmp_checkpoint
