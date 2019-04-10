import torch
import numpy as np


class ImageDatabase(torch.utils.data.Dataset):
    """
    Dataset for IMDB used in Pythia
    General format that we have standardize follows:
    {
        metadata: {
            'version': x
        },
        data: [
            {
                'id': DATASET_SET_ID,
                'set_folder': <directory>,
                'feature_path': <file_path>,
                'info': {
                    // Extra information
                    'questions_tokens': [],
                    'answer_tokens': []
                }
            }
        ]
    }
    """
    def __init__(self, imdb_path):
        super().__init__()

        if not imdb_path.endswith(".npy"):
            raise ValueError("Unknown file format for imdb")

        self.db = np.load(imdb_path)
        self.start_idx = 0

        if type(self.db) == dict:
            self.metadata = self.db.get('metadata', {})
            self.data = self.db.get('data', [])
        else:
            # TODO: Deprecate support for this
            self.metadata = {'version': 1}
            self.data = self.db
            # Handle old imdb support
            if "image_id" not in self.data[0]:
                self.start_idx = 1

        if len(self.data) == 0:
            self.data = self.db

    def __len__(self):
        return len(self.data) - self.start_idx

    def __getitem__(self, idx):
        return self.data[idx + self.start_idx]

    def get_version(self):
        return self.metadata.get('version', None)
