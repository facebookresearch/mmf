# Copyright (c) Facebook, Inc. and its affiliates.
class BaseDatasetBuilder:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load(self, dataset_type, config, *args, **kwargs):
        """WARNING: DO NOT OVERRIDE in child class. Instead override '_load'.
        Main load function use by Pythia. This will internally call '_load'
        function.

        Parameters
        ----------
        **kwargs : dict
            Dictionary containing parameters necessary to load the dataset

        Returns
        -------
        dataset: torch.utils.Dataset
            Dataset containing data to be trained on

        """
        dataset = self._load(dataset_type, config, *args, **kwargs)
        dataset.init_processors()
        dataset.try_fast_read()
        return dataset

    def _load(self, dataset_type, config, *args, **kwargs):
        """
        This is used to prepare the dataset and load it from a path.
        Override this method in your child dataset builder class.
        """
        raise NotImplementedError("This dataset builder doesn't implement a "
                                  "load method")

    def build(self, dataset_type, config, *args, **kwargs):
        """WARNING: DO NOT OVERRIDE in child class. Instead override '_build'.
        Similar to load function, used by Pythia to build a dataset for first
        time when it is not available. This internally calls '_build' function.
        Override that function in your child class.

        Parameters
        ----------
        **kwargs : dict
            Contains dictionary of parameters essential for building a dataset

        """
        # TODO: Once we start building we will do some preprocessing for folder
        # structure and other things here
        self._build(dataset_type, config, *args, **kwargs)

    def _build(self, dataset_type, config, *args, **kwargs):
        """
        This is used to build a dataset first time.
        Override this method in your child dataset builder class.
        """
        raise NotImplementedError("This dataset builder doesn't implement a "
                                  "build method")
