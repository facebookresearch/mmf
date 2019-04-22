# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from torch.utils.data import Dataset

from pythia.common.registry import registry


class MultiTask(Dataset):
    def __init__(self, dataset_type, config):
        super(MultiTask, self).__init__()

        self.config = config
        self.dataset_type = dataset_type

        self.task_names = map(lambda x: x.strip(), self.config["tasks"].split(","))

        self.tasks = []
        self.tasks_lens = []

        for task_name in self.task_names:
            task_class = registry.get_task_class(task_name)
            if task_class is None:
                print("[Error] %s not present in our mapping" % task_name)
                return

            if task_name not in self.config["task_attributes"]:
                print(
                    "[Error] No attributes present for task %s in config."
                    " Skipping" % task_name
                )

            task_attributes = self.config["task_attributes"][task_name]
            task_attributes["dataset_type"] = self.dataset_type

            task = task_class()
            task.load(**task_attributes)

            self.tasks.append(task)
            self.tasks_lens.append(len(task))

        self.task_probabilities = [1 for _ in self.tasks]

        self.num_tasks = len(self.tasks)

        training_parameters = self.config["training_parameters"]
        if training_parameters["task_size_proportional_sampling"]:
            self.task_probabilities = self.tasks_lens[:]
            len_sum = sum(self.tasks_lens)
            self.task_probabilities = [
                prob / len_sum for prob in self.task_probabilities
            ]

        self.change_task()

    def change_task(self):
        self.selected_task = np.random.choice(
            self.num_tasks, 1, p=self.task_probabilities
        )[0]
        self.chosen_task = self.tasks[self.selected_task]
        self.chosen_task.change_dataset()

    def get_tasks(self):
        return self.tasks

    def verbose_dump(self, *args):
        self.chosen_task.verbose_dump(*args)

    def __len__(self):
        return sum(self.tasks_lens)

    def __getitem__(self, idx):
        idx = idx % self.tasks_lens[self.selected_task]
        item = self.chosen_task[idx]

        return item

    def update_registry_for_model(self, config):
        for task in self.tasks:
            task.update_registry_for_model(config)

    def prepare_batch(self, batch):
        return self.chosen_task.prepare_batch(batch)

    def init_args(self, parser):
        for task in self.tasks:
            task.init_args(parser)

    def clean_config(self, config):
        for task in self.tasks:
            task.clean_config(config)

        return config
