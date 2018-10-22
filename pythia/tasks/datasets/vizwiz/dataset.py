from pythia.tasks.datasets.vqa2.dataset import VQA2Dataset


class VizWizDataset(VQA2Dataset):
    def __init__(self, imdb_file, image_feat_directories, verbose=False,
                 **data_params):
        super(VizWizDataset, self).__init__(imdb_file, image_feat_directories,
                                            verbose, **data_params)
        self.max_valid_answer_length = 11
