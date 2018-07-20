from torch.utils.data import ConcatDataset


class vqa_concate_dataset(ConcatDataset):
    def __init__(self, datasets):
        super(vqa_concate_dataset, self).__init__(datasets)
        self.vocab_dict = datasets[0].vocab_dict
        self.answer_dict = datasets[0].answer_dict
