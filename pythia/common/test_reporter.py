import os
import json

from torch.utils.data import Dataset, DataLoader

from pythia.utils.general import ckpt_name_from_core_args, \
                                 foldername_from_config_override
from pythia.core.registry import registry
from pythia.utils.timer import Timer


class TestReporter(Dataset):
    def __init__(self, multi_task_instance):
        self.test_task = multi_task_instance
        self.config = registry.get('config')
        self.writer = registry.get('writer')
        self.report = []
        self.timer = Timer()
        self.training_parameters = self.config['training_parameters']
        self.num_workers = self.training_parameters['num_workers']
        self.batch_size = self.training_parameters['batch_size']
        self.report_folder_arg = self.config.get('report_folder', None)
        self.experiment_name = self.training_parameters.get('experiment_name',
                                                            "")

        self.datasets = []

        for task in self.test_task.get_tasks():
            for dataset in task.get_datasets():
                self.datasets.append(dataset)

        self.current_dataset_idx = -1
        self.current_dataset = self.datasets[self.current_dataset_idx]

        self.save_dir = self.config.get('save_dir', "./save")
        self.report_folder = ckpt_name_from_core_args(self.config)
        self.report_folder += foldername_from_config_override(self.config)

        self.report_folder = os.path.join(self.save_dir, self.report_folder)
        self.report_folder = os.path.join(self.report_folder, "reports")

        if self.report_folder_arg is not None:
            self.report_folder = self.report_folder_arg

        if not os.path.exists(self.report_folder):
            os.makedirs(self.report_folder)

    def next_dataset(self):
        if self.current_dataset_idx >= 0:
            self.flush_report()

        self.current_dataset_idx += 1

        if self.current_dataset_idx == len(self.datasets):
            return False
        else:
            self.current_dataset = self.datasets[self.current_dataset_idx]
            self.writer.write("Predicting for " + self.current_dataset.name)
            return True

    def flush_report(self):
        name = self.current_dataset.name
        time_format = "%Y-%m-%dT%H:%M:%S"
        time = self.timer.get_time_hhmmss(None, time_format)

        filename = name + "_"

        if len(self.experiment_name) > 0:
            filename += self.experiment_name + "_"

        filename += time + ".json"
        filepath = os.path.join(self.report_folder, filename)

        with open(filepath, 'w') as f:
            json.dump(self.report, f)

        self.writer.write("Wrote evalai predictions for %s to %s" %
                          (name, os.path.abspath(filepath)))
        self.report = []

    def get_dataloader(self):
        return DataLoader(
            dataset=self.current_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def prepare_batch(self, batch):
        return self.current_dataset.prepare_batch(batch)

    def __len__(self):
        return len(self.current_dataset)

    def __getitem__(self, idx):
        return self.current_dataset[idx]

    def add_to_report(self, batch, answers):
        results = self.current_dataset.format_for_evalai(batch, answers)

        self.report = self.report + results
