class DatasetBuilder:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load(self, **opts):
        """
        This is used to prepare the dataset and load it from a path.
        Override this method in your child dataset builder class.
        """
        raise NotImplementedError("This dataset builder doesn't implement a "
                                  "load method")

    def build(self, **opts):
        """
        This is used to build a dataset first time.
        Override this method in your child dataset builder class.
        """
        raise NotImplementedError("This dataset builder doesn't implement a "
                                  "build method")
