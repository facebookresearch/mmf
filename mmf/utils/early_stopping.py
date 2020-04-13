# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch

from mmf.utils.distributed import is_master


class EarlyStopping:
    """
    Provides early stopping functionality. Keeps track of model metrics,
    and if it doesn't improve over time restores last best performing
    parameters.
    """

    def __init__(
        self,
        model,
        checkpoint_instance,
        monitored_metric="total_loss",
        patience=1000,
        minimize=False,
        should_stop=True,
    ):
        self.minimize = minimize
        self.patience = patience
        self.model = model
        self.checkpoint = checkpoint_instance
        self.monitored_metric = monitored_metric

        if "val" not in self.monitored_metric:
            self.monitored_metric = "val/{}".format(self.monitored_metric)

        self.best_monitored_value = -np.inf if not minimize else np.inf
        self.best_monitored_iteration = 0
        self.best_monitored_update = 0
        self.should_stop = should_stop
        self.activated = False
        self.metric = self.monitored_metric

    def __call__(self, update, iteration, meter):
        """
        Method to be called everytime you need to check whether to
        early stop or not
        Arguments:
            update {number}: Current update number
            iteration {number}: Current iteration number
        Returns:
            bool -- Tells whether early stopping occurred or not
        """
        if not is_master():
            return False

        value = meter.meters.get(self.monitored_metric, None)
        if value is None:
            raise ValueError(
                "Metric used for early stopping ({}) is not "
                "present in meter.".format(self.monitored_metric)
            )

        value = value.global_avg

        if isinstance(value, torch.Tensor):
            value = value.item()

        if (self.minimize and value < self.best_monitored_value) or (
            not self.minimize and value > self.best_monitored_value
        ):
            self.best_monitored_value = value
            self.best_monitored_iteration = iteration
            self.best_monitored_update = update
            self.checkpoint.save(update, iteration, update_best=True)

        elif self.best_monitored_update + self.patience < update:
            self.activated = True
            if self.should_stop is True:
                self.checkpoint.restore()
                self.checkpoint.finalize()
                return True
            else:
                return False
        else:
            self.checkpoint.save(update, iteration, update_best=False)

        return False

    def is_activated(self):
        return self.activated

    def init_from_checkpoint(self, load):
        if "best_iteration" in load:
            self.best_monitored_iteration = load["best_iteration"]

        if "best_metric_value" in load:
            self.best_monitored_value = load["best_metric_value"]

    def get_info(self):
        return {
            "best_update": self.best_monitored_update,
            "best_iteration": self.best_monitored_iteration,
            "best_{}".format(self.metric): "{:.6f}".format(self.best_monitored_value),
        }
