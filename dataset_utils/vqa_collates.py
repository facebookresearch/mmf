from torch.utils.data.dataloader import default_collate
import numpy as np


def filter_unk_collate(batch):
    batch = list(filter(lambda x: np.sum(x['ans_scores']) >0, batch))
    return default_collate(batch)


