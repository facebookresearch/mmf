import os

from pythia.core.tasks.datasets.features_dataset import FeaturesDataset
from pythia.core.tasks.datasets.utils.feature_readers import FeatureReader


class COCOFeaturesDataset(FeaturesDataset):
    def __init__(self, **kwargs):
        super(COCOFeaturesDataset, self).__init__()
        self.feature_readers = []
        self.feature_dict = {}

        self.fast_read = kwargs['fast_read']

        for image_feature_dir in kwargs['image_feature_dirs']:
            feature_reader = FeatureReader(
                                    base_path=image_feature_dir,
                                    channel_first=kwargs['channel_first'],
                                    max_bboxes=kwargs.get('max_bboxes', None),
                                    image_feature=kwargs.get('image_feature',
                                                             None))
            self.feature_readers.append(feature_reader)

        self.dataset_type = kwargs.get('dataset_type', None)
        self.imdb = kwargs['imdb']
        self.kwargs = kwargs
        self.should_return_info = kwargs.get('return_info', False)

        if self.fast_read:
            for feat_file in os.listdir(image_feature_dir):
                features, info = self._read_features_and_info(feat_file)
                self.feature_dict[feat_file] = (features, info)

    def _read_features_and_info(self, feat_file):
        features = []
        infos = []
        for feature_reader in self.feature_readers:
            feature, info = feature_reader.read(feat_file)
            features.append(feature)
            infos.append(info)

        if not self.should_return_info:
            infos = None
        return features, infos

    def _get_image_features_and_info(self, feat_file):
        image_feats, infos = self.feature_dict.get(feat_file, (None, None))

        if image_feats is None:
            image_feats, infos = self._read_features_and_info(feat_file)

        # TODO: Remove after standardization
        # https://github.com/facebookresearch/pythia/blob/master/dataset_utils/dataSet.py#L226
        return image_feats, infos

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        image_info = self.imdb[idx]
        image_file_name = image_info['feature_path']

        image_features, infos = \
            self._get_image_features_and_info(image_file_name)

        item = {}
        for idx, image_feature in enumerate(image_features):
            item["image_feature_%s" % idx] = image_feature
            if infos is not None:
                item["image_info_%s" % idx] = infos[idx]

        return item
