# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from torch.utils.data import ConcatDataset


class vqa_concate_dataset(ConcatDataset):
    def __init__(self, datasets):
        super(vqa_concate_dataset, self).__init__(datasets)
        self.vocab_dict = datasets[0].vocab_dict
        self.answer_dict = datasets[0].answer_dict
