# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Optional

from mmf.utils.distributed import is_master


try:
    import wandb
except ImportError:
    wandb = None


class WandbLogger:
    r"""
    Log using `Weights and Biases`.

    Args:
        name: Display name for the run.
        save_dir: Path where data is saved (./save/logs/wandb/ by default).
        project: Display name for the project.
        **init_kwargs: Arguments passed to :func:`wandb.init`.

    Raises:
        ImportError: If wandb package is not installed.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        project: Optional[str] = None,
        **init_kwargs,
    ):

        if wandb is None:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )
        else:
            self._wandb = wandb

        self._wandb_init = dict(name=name, project=project, dir=save_dir)

        self._wandb_init.update(**init_kwargs)

        self._is_master = is_master()

        self.setup()

    def setup(self):
        """
        Setup `Weights and Biases` for logging.
        """
        if self._is_master:

            if self._wandb.run is None:
                self._wandb.init(**self._wandb_init)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("trainer/global_step")
                self._wandb.define_metric(
                    "*", step_metric="trainer/global_step", step_sync=True
                )

    def __del__(self):
        if getattr(self, "_wandb", None) is not None:
            self._wandb.finish()

    def _should_log_wandb(self):
        if self._wandb is None or not self._is_master:
            return False
        else:
            return True

    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log the monitored metrics to the wand dashboard.

        Args:
            metrics (Dict[str, float]): [description]
        """
        if not self._should_log_wandb():
            return

        self._wandb.log(metrics)
