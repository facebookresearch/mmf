import importlib

from torch.utils.data import DataLoader


class TaskLoader:
    def __init__(self, config):
        self.config = config
        return

    def load_dataset(self):
        base_task_path = "pythia.tasks"
        tasks_module = importlib.import_module(base_task_path)
        task_class = getattr(tasks_module, self.config['task'])
        task_name = self.config['task']

        self.train_dataset = task_class(task_name, 'train')
        self.dev_dataset = task_class(task_name, 'dev')
        self.test_dataset = task_class(task_name, 'test')

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

    def update_config_for_model(self, config):
        self.train_dataset.update_config_for_model(config)
        self.dev_dataset.update_config_for_model(config)
        self.test_dataset.update_config_for_model(config)
