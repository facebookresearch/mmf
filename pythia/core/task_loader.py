import os
import yaml

from torch.utils.data import DataLoader

from pythia.core.tasks import MultiTask
from pythia.utils.general import nested_dict_update


class TaskLoader:
    def __init__(self, config):
        self.config = config

    def load_task(self):
        self.train_task = MultiTask('train', self.config)
        self.dev_task = MultiTask('dev', self.config)
        self.test_task = MultiTask('test', self.config)

        self.mapping = {
            'train': self.train_task,
            'dev': self.dev_task,
            'test': self.test_task
        }

    def load_config(self):

        task_names = map(lambda x: x.strip(),
                         self.config['tasks'].split(","))

        self.task_config = {}

        for task in task_names:
            current_task_config = self._load_task_config(task)
            self.task_config = nested_dict_update(self.task_config,
                                                  current_task_config)

    def get_config(self):
        return self.task_config

    def _load_task_config(self, task_name):
        directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(directory, '..', 'tasks',
                                   task_name, 'config.yml')
        task_config = {}
        if not os.path.exists(config_path):
            print("[Warning] No config present for task %s" %
                  self.config['task'])
            return

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
                                       num_workers=num_workers)
        self.train_loader.dataset_type = 'train'

        self.dev_loader = DataLoader(dataset=self.dev_task,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)
        self.dev_loader.dataset_type = 'dev'

        self.test_loader = DataLoader(dataset=self.test_task,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)
        self.test_loader.dataset_type = 'test'

        self.use_cuda = self.config['use_cuda']

    def update_config_for_model(self, config):
        self.train_task.update_config_for_model(config)
        self.dev_task.update_config_for_model(config)
        self.test_task.update_config_for_model(config)

    def clean_config(self, config):
        self.train_task.clean_config(config)
        self.dev_task.clean_config(config)
        self.test_task.clean_config(config)

    def report_metrics(self, dataset_type, loss, extra_info=None,
                       should_print=True):
        if not self.config['should_log']:
            return
        # TODO: Complete this by calling child report metrics
        task = self.mapping[dataset_type]
        task.report_metrics(loss, extra_info, should_print)

    def calculate_loss(self, dataset_type, output, expected_output):
        return self.mapping[dataset_type].calculate_loss(output,
                                                         expected_output)

    def prepare_batch(self, dataset_type, batch):
        return self.mapping[dataset_type].prepare_batch(batch)

    def reset_meters(self, dataset_type):
        self.mapping[dataset_type].reset_meters()
