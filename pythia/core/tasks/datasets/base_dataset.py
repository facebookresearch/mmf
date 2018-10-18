from torch.utils.data.dataset import Dataset

from pythia.core.losses import Loss
from pythia.core.meter import Meter
from pythia.core.registry import Registry


class BaseDataset(Dataset):
    def __init__(self, name, config={}):
        super(BaseDataset, self).__init__()
        self.config = config
        self.name = name
        self._init_loss_and_metrics(config)
        self.writer = Registry.get('writer')

    def _init_loss_and_metrics(self, config):
        task_metrics = config.get('metrics', [])
        if isinstance(task_metrics, str):
            task_metrics = task_metrics.split(',')

        self.meter = Meter(config['dataset_type'], task_metrics)
        self.loss_fn = Loss(config['loss'])

    def calculate_loss(self, output, expected_output):
        self.meter(output, expected_output)

        self.last_loss = self.loss_fn(output, expected_output)

        return self.last_loss

    def report_metrics(self, loss=None, extra_info=None,
                       should_print=True):
        if not self.config['should_log']:
            return

        if loss is None:
            loss = self.last_loss
        if should_print:
            log_string = self.meter.get_log_string(loss)

            if extra_info is not None:
                log_string += " " + extra_info
            self.writer.write(log_string)

        dataset_type = self.meter.get_task_type()

        scalars = {}
        for i in range(len(self.meter.meter_types)):
            meter_type = self.meter.meter_types[i]
            value = self.meter.meter_values[i]

            key = "%s_%s_%s" % (self.name, dataset_type, meter_type)
            scalars[key] = value
        self.writer.add_scalars(scalars, Registry.get('current_iteration'))
