import torch


class Flickr30kAnnotationDatabase(torch.utils.data.Dataset):
    """
    Dataset for Flickr30k Annotations used in MMF
    """

    def __init__(self, config, splits_path, ann_path, *args, **kwargs):
        super().__init__()
        self.config = config
        self._load_annotation_db(splits_path, ann_path)

    def _load_annotation_db(self, splits_path, ann_path):
        data = []

        with open(splits_path, 'r') as f:
            valid_annotations = f.read().splitlines()

        with open(ann_path, 'r') as f:
            lines = f.read().splitlines()

            # Extract all annotations
            annotations = {}
            for line in lines:
                parsed_line = line.replace('\t', '').split('#')
                image_id = parsed_line[0]
                if image_id[:-4] in valid_annotations:
                    sentence = parsed_line[1][1:-2]
                    if image_id not in annotations:
                        annotations[image_id] = [sentence]
                    else:
                        annotations[image_id].append(sentence)

            # Build annotations
            for key, value in annotations.items():
                data.append(
                    {
                        "image_path": key,
                        "sentences": value,
                    }
                )

        self.data = data

        if len(self.data) == 0:
            raise RuntimeError("Dataset is empty")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
