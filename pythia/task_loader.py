import importlib
import os
import yaml

from torch.utils.data import DataLoader

from pythia.utils.meter import Meter


class TaskLoader:
    def __init__(self, config):
        self.config = config
        return

    def load_dataset(self):
        base_task_path = "pythia.tasks"
        tasks_module = importlib.import_module(base_task_path)
        task_class = getattr(tasks_module, self.config['task'])
        self.task_name = self.config['task']

        self.train_dataset = task_class(self.task_name, 'train')
        self.dev_dataset = task_class(self.task_name, 'dev')
        self.test_dataset = task_class(self.task_name, 'test')

        self.train_dataset.load(self.config['task_attributes'])
        self.dev_dataset.load(self.config['task_attributes'])
        self.test_dataset.load(self.config['task_attributes'])

        self.make_meters()

    def load_config(self):
        directory = os.path.dirname(os.path.abspath(__file__))

        config_path = os.path.join(directory, 'tasks', self.task_name,
                                   'config.yml')

        if not os.path.exists(config_path):
            return

        self.task_config = {}
        with open(config_path, 'r') as f:
            try:
                self.task_config = yaml.load(f)
            except yaml.YAMLError as err:
                print("[Error] Task %s's config yaml error" % self.task_name,
                      err)

    def make_dataloaders(self):
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.config['batch_size'],
                                       shuffle=True,
                                       num_workers=self.config['num_workers'])

        self.dev_loader = DataLoader(dataset=self.val_dataset,
                                     batch_size=self.config['batch_size'],
                                     shuffle=True,
                                     num_workers=self.config['num_workers'])
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.config['batch_size'],
                                      shuffle=True,
                                      num_workers=self.config['num_workers'])

    def make_meters(self):
        task_metrics = self.config['task_attributes']['metrics']
        task_metrics = task_metrics.split(',')

        self.train_meter = Meter('train', self.config, task_metrics)
        self.dev_meter = Meter('dev', self.config, task_metrics)
        self.test_meter = Meter('test', self.config, task_metrics)

    def update_config_for_model(self, config):
        self.train_dataset.update_config_for_model(config)
        self.dev_dataset.update_config_for_model(config)
        self.test_dataset.update_config_for_model(config)

    def report_metrics(self, writer, meter, loss,
                       iteration, should_print=True):
        if not self.config['should_log']:
            return

        dataset_type = meter.get_dataset_type()

        if should_print:
            log_string = meter.get_log_string()
            writer.write(log_string)

        scalars = {}
        for i in range(meter.meter_types):
            meter_type = meter.meter_types[i]
            value = meter.avg_meter_values[i]
            value /= meter.iteration_count

            key = "%s_%s" % (dataset_type, meter_type)
            scalars[key] = value

        writer.add_scalars(scalars)
