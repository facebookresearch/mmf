import os
import yaml

from torch.utils.data import DataLoader

from pythia.tasks import MultiTask
from .batch_collator import BatchCollator
from .test_reporter import TestReporter


class TaskLoader:
    def __init__(self, config):
        self.config = config

    def load_task(self):
        self.train_task = MultiTask('train', self.config)
        self.val_task = MultiTask('val', self.config)
        self.test_task = MultiTask('test', self.config)

        self.mapping = {
            'train': self.train_task,
            'val': self.val_task,
            'test': self.test_task
        }

        self.test_reporter = None
        if self.config.training_parameters.evalai_predict is True:
            self.test_reporter = TestReporter(self.test_task)

    def get_config(self):
        return self.task_config

    def _load_task_config(self, task_name):
        directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(directory, '..', 'tasks',
                                   task_name, 'config.yml')
        task_config = {}
        if not os.path.exists(config_path):
            print("[Warning] No config present for task %s" %
                  task_name)
            return {}

        with open(config_path, 'r') as f:
            try:
                task_config = yaml.load(f)
            except yaml.YAMLError as err:
                print("[Error] Task %s's config yaml error" % self.task_name,
                      err)

        return task_config

    def make_dataloaders(self):
        training_parameters = self.config['training_parameters']
        batch_size = training_parameters['batch_size']
        num_workers = training_parameters['num_workers']

        self.train_loader = DataLoader(dataset=self.train_task,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       collate_fn=BatchCollator(),
                                       num_workers=num_workers)
        self.train_loader.dataset_type = 'train'

        self.val_loader = DataLoader(dataset=self.val_task,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     collate_fn=BatchCollator(),
                                     num_workers=num_workers)
        self.val_loader.dataset_type = 'val'

        self.test_loader = DataLoader(dataset=self.test_task,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      collate_fn=BatchCollator(),
                                      num_workers=num_workers)
        self.test_loader.dataset_type = 'test'

        self.use_cuda = "cuda" in self.config.training_parameters.device

    def update_registry_for_model(self, config):
        self.train_task.update_registry_for_model(config)
        self.val_task.update_registry_for_model(config)
        self.test_task.update_registry_for_model(config)

    def clean_config(self, config):
        self.train_task.clean_config(config)
        self.val_task.clean_config(config)
        self.test_task.clean_config(config)

    def report_metrics(self, dataset_type, *args, **kwargs):
        if not self.config.should_log:
            return
        # TODO: Complete this by calling child report metrics
        task = self.mapping[dataset_type]
        task.report_metrics(*args, **kwargs)

    def calculate_loss_and_metrics(self, dataset_type, *args, **kwargs):
        task = self.mapping[dataset_type]
        return task.calculate_loss_and_metrics(*args, **kwargs)

    def prepare_batch(self, dataset_type, batch):
        return self.mapping[dataset_type].prepare_batch(batch)

    def reset_meters(self, dataset_type):
        self.mapping[dataset_type].reset_meters()

    def verbose_dump(self, dataset_type, *args, **kwargs):
        if self.config.training_parameters.verbose_dump:
            self.mapping[dataset_type].verbose_dump(*args, **kwargs)
