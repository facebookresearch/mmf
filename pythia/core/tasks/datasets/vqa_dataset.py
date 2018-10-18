from .image_dataset import ImageDataset


class VQADataset(ImageDataset):
    def __init__(self, vqa_db_path):
        super(VQADataset, self).__init__(self, vqa_db_path)

    def __getitem__(self, idx):
        item = self.db[idx]

        return item['feature_path'], item['info']['question_tokens'], \
            item['info']['answer_tokens']
