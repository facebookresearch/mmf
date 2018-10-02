import importlib

from torch.utils.data import Dataset


class BaseTask(Dataset):
    def __init__(self, task_name, datasets):
        self.task_name = task_name
        self.given_datasets = datasets

    def load(self, opts):
        self.datasets = []
        available_datasets = self.get_available_datasets()
        dataset_module = importlib.import_module('pythia.tasks.datasets')

        self.total_length = 0
        self.per_dataset_lengths = []

        for dataset in self.given_datasets:
            if dataset in available_datasets:
                dataset_class = getattr(dataset_module, dataset)
                dataset_instance = dataset_class(opts)
                self.datasets.append(dataset_instance)
                self.per_dataset_lengths.append(len(dataset_instance))
                self.total_length += len(dataset_instance)

    def get_available_datasets(self):
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

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        raise NotImplementedError("This task doesn't implement getitem method")

    def prepare_batch(self, batch, use_cuda=False):
        '''
        Override in your child class

        Prepare batch for passing to model. Whatever returned from here will
        be directly passed to model's forward function
        '''
        return batch

    def update_config_for_model(self, config):
        '''
        Use this if there is some specific configuration required by model
        which must be inferred at runtime.
        '''
        raise NotImplementedError("This task doesn't implement config"
                                  " update method")

    def init_args(self, parser):
        parser.add_argument_group('General Task Arguments')
        parser.add_argument('-tp', '--size_proportional_sampling', type=bool,
                            default=0, help="Pass if you want to sample from"
                            " dataset according to its size. Default: Equal "
                            " weighted sampling")

        self._init_args(parser)

    def _init_args(self, parser):
        """Override this function to update parser in your child task class.

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

        raise NotImplementedError("This task doesn't implement args "
                                  " initialization method")

    def clean_config(self, config):
        '''
        Use this in case you want to clean the config you updated earlier
        in update_config_for_model
        '''
        return
