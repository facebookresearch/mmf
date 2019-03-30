class DatasetBuilder:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load(self, **opts):
        """WARNING: DO NOT OVERRIDE in child class. Instead override '_load'.
        Main load function use by Pythia. This will internally call '_load'
        function.

        Parameters
        ----------
        **opts : dict
            Dictionary containing parameters necessary to load the dataset

        Returns
        -------
        dataset: torch.utils.Dataset
            Dataset containing data to be trained on

        """
        dataset = self._load(**opts)
        dataset.init_loss_and_metrics(opts)
        return dataset

    def _load(self, **opts):
        """
        This is used to prepare the dataset and load it from a path.
        Override this method in your child dataset builder class.
        """
        raise NotImplementedError("This dataset builder doesn't implement a "
                                  "load method")

    def build(self, **opts):
        """WARNING: DO NOT OVERRIDE in child class. Instead override '_build'.
        Similar to load function, used by Pythia to build a dataset for first
        time when it is not available. This internally calls '_build' function.
        Override that function in your child class.

        Parameters
        ----------
        **opts : dict
            Contains dictionary of parameters essential for building a dataset

        """
        # TODO: Once we start building we will do some preprocessing for folder
        # structure and other things here
        self._build(**opts)

    def _build(self, **opts):
        """
        This is used to build a dataset first time.
        Override this method in your child dataset builder class.
        """
        raise NotImplementedError("This dataset builder doesn't implement a "
                                  "build method")
