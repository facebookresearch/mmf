import numpy as np

from torch.utils.data import Dataset

from pythia.core.registry import Registry


class BaseTask(Dataset):
    def __init__(self, task_name):
        super(BaseTask, self).__init__()
        self.task_name = task_name
        self.writer = Registry.get('writer')

    def _process_datasets(self):
        if 'datasets' not in self.opts:
            self.writer.write("No datasets attribute present for task: %s."
                              " Defaulting to all" % (self.task_name),
                              'warning')
            datasets = "all"
        else:
            datasets = self.opts['datasets']

        if datasets is None or datasets == "all":
            datasets = self._get_available_datasets()

        if type(datasets) == str:
            datasets = list(map(lambda x: x.strip(),
                            datasets.split(",")))

        if len(datasets) == 0 and datasets[0] == "all":
            datasets = self._get_available_datasets()

        self.given_datasets = datasets

    def load(self, **opts):
        self.opts = opts
        self.use_cuda = Registry.get('config')['use_cuda']
        self._process_datasets()

        self.datasets = []
        self.builders = []
        available_datasets = self._get_available_datasets()

        self.total_length = 0
        self.per_dataset_lengths = []
        self.num_datasets = 0

        for dataset in self.given_datasets:
            if dataset in available_datasets:
                builder_class = Registry.get_builder_class(dataset)

                if builder_class is None:
                    print("No builder class found for %s." % dataset)
                    continue
                builder_instance = builder_class()

                attributes = self.opts['dataset_attributes'][dataset]
                attributes['dataset_type'] = self.opts.get('dataset_type',
                                                           "train")
                builder_instance.build(**attributes)
                dataset_instance = builder_instance.load(**attributes)

                self.builders.append(builder_instance)
                self.datasets.append(dataset_instance)
                self.per_dataset_lengths.append(len(dataset_instance))
                self.total_length += len(dataset_instance)
            else:
                print("Dataset %s is not a valid dataset for task %s. Skipping"
                      % (dataset, self.task_name))

        self.num_datasets = len(self.datasets)
        self.dataset_probablities = [1 for _ in range(self.num_datasets)]
        sampling = self.opts.get('dataset_size_proportional_sampling', None)

        if sampling is True:
            self.dataset_probablities = self.per_dataset_lengths[:]
            self.dataset_probablities = [prob / self.total_length
                                         for prob in
                                         self.dataset_probablities]

        self.change_dataset()

    def _get_available_datasets(self):
        """Set available datasets for this task here.
        Override in your child task class
        Temporary solution, later we will use decorators to easily register
        datasets with a task

        Returns
        -------
        type
            List - List of available datasets for this particular task
        """
        return []

    def get_datasets(self):
        return self.datasets

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        idx = idx % self.per_dataset_lengths[self.dataset_choice]

        item = self.chosen_dataset[idx]

        return self._preprocess_item(item)

    def change_dataset(self):
        self.dataset_choice = np.random.choice(self.num_datasets, 1,
                                               p=self.dataset_probablities)[0]
        self.chosen_dataset = self.datasets[self.dataset_choice]

    def calculate_loss(self, output, expected_output, info):
        return self.chosen_dataset.calculate_loss(output, expected_output,
                                                  info)

    def report_metrics(self, loss, extra_info=None, should_print=True):
        self.chosen_dataset.report_metrics(loss, extra_info, should_print)

    def reset_meters(self):
        self.chosen_dataset.reset_meters()

    def prepare_batch(self, batch):
        return self.chosen_dataset.prepare_batch(batch)

    def _preprocess_item(self, item):
        """Preprocess an item to be returned from __getitem__.
        Override in your child task class, so you have control on what you are
        returning

        Parameters
        ----------
        item : Object
            Object returned by a particular dataset

        Returns
        -------
        Object:
            Preprocessed item
        """
        raise NotImplementedError("This task doesn't implement preprocess_item"
                                  " method")

    def update_config_for_model(self, config):
        """
        Use this if there is some specific configuration required by model
        which must be inferred at runtime.
        """
        for builder in self.builders:
            builder.update_config_for_model(config)

    def init_args(self, parser):
        parser.add_argument_group('General Task Arguments')
        parser.add_argument('-dsp', '--dataset_size_proportional_sampling',
                            type=bool, default=0,
                            help="Pass if you want to sample from"
                            " dataset according to its size. Default: Equal "
                            " weighted sampling")

        # TODO: Figure out later if we want to init args from datasets
        # self._init_args(parser)

    def _init_args(self, parser):
        """Override this function to add extra parameters to
        parser in your child task class.

        Parameters
        ----------
        parser : ArgumentParser
            Original parser object passed from the higher level classes like
            trainer

        Returns
        -------
        type
            Description of returned object.

        """
        for builder in self.builders:
            builder.init_args(parser)

    def clean_config(self, config):
        """
        Override this in case you want to clean the config you updated earlier
        in update_config_for_model
        """
        return config
