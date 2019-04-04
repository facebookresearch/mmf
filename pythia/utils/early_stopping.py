import numpy as np

from pythia.common.registry import registry


class EarlyStopping:
    """
    Provides early stopping functionality. Keeps track of model metrics,
    and if it doesn't improve over time restores last best performing
    parameters.

    Use monitored_metric = -1 for monitoring based on loss otherwise use
    array index of your metric in metrics section of your config
    """

    def __init__(self, model, checkpoint_instance, monitored_metric=-1,
                 patience=1000, minimize=False, should_stop=True):
        self.minimize = minimize
        self.patience = patience
        self.model = model
        self.checkpoint = checkpoint_instance
        self.monitored_metric = monitored_metric
        self.best_monitored_metric = -np.inf if not minimize else np.inf
        self.best_monitored_iteration = 0
        self.should_stop = should_stop
        self.activated = False
        self.metric = self.monitored_metric

    def __call__(self, iteration):
        """
        Method to be called everytime you need to check whether to
        early stop or not
        Arguments:
            iteration {number}: Current iteration number
        Returns:
            bool -- Tells whether early stopping occurred or not
        """
        value = registry.get('metrics.dev.%s' % self.monitored_metric)
        if hasattr(value, 'data'):
            value = value.data.item()

        if (self.minimize and value < self.best_monitored_metric) or \
                (not self.minimize and value > self.best_monitored_metric):
            self.best_monitored_metric = value
            self.best_monitored_iteration = iteration
            self.checkpoint.save(iteration, update_best=True)

        elif self.best_monitored_iteration + self.patience < iteration:
            self.activated = True
            if self.should_stop is True:
                self.checkpoint.restore()
                self.checkpoint.finalize()
                return True
            else:
                return False
        else:
            self.checkpoint.save(iteration, update_best=False)

        return False

    def is_activated(self):
        return self.activated

    def init_from_checkpoint(self, load):
        self.best_monitored_iteration = load['best_iteration']
        self.best_monitored_metric = load['best_metric']

    def get_info(self):
        return "Best %s: %0.6f, Best iteration: %d" % \
                (self.metric, self.best_monitored_metric,
                 self.best_monitored_iteration)
