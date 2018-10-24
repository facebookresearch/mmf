from pythia.tasks.datasets.vqa2.dataset import VQA2Dataset


class VizWizDataset(VQA2Dataset):
    def __init__(self, imdb_file, image_feat_directories, verbose=False,
                 **data_params):
        super(VizWizDataset, self).__init__(imdb_file, image_feat_directories,
                                            verbose, **data_params)

        self.name = 'vizwiz'
        if data_params['copy_included']:
            self.max_valid_answer_length = 11

    def __getitem__(self, idx):
        sample = super(VizWizDataset, self).__getitem__(idx)
        idx = self.first_element_idx + idx
        image = self.imdb[idx]
        sample['image_id'] = image['image_name']
        return sample

    def format_for_evalai(self, batch, answers):
        answers = answers.argmax(dim=1)

        predictions = []

        for idx, image_id in enumerate(batch['image_id']):
            answer = self.answer_dict.idx2word(answers[idx])
            # 'COCO_vizwiz_test_000000020255' -> 'VizWiz_test_000000020255.jpg'
            predictions.append({
                'image': "_".join(["VizWiz"] + image_id.split("_")[2:])
                         + ".jpg",
                'answer': answer
            })

        return predictions
