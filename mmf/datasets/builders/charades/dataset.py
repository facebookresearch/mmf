import os

import PIL
import torch
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.builders.charades._utils import (
    CharadesVideoClips,
    img2gif,
    make_charades_df,
)
from mmf.utils.distributed import byte_tensor_to_object, object_to_byte_tensor
from mmf.utils.file_io import PathManager


class CharadesDataset(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__("charades", config, dataset_type)
        self.imdb_file_index = imdb_file_index
        self.load_df()
        self.length = len(self.video_clips)
        self.audio_processor = None
        self.video_processor = None
        self.video_train_processor = None
        self.video_test_processor = None
        self.prediction_threshold = self.config.get("prediction_threshold", 0.5)
        # Some pickling related issues can be resolved by loading it at runtime
        # optionally, uncomment next line if you face those issues.
        # self.video_clips = []

    def init_processors(self):
        super().init_processors()
        self.set_processors()

    def load_df(self):
        dataset_type = self.dataset_type
        imdb_file_index = self.imdb_file_index
        config = self.config
        csv_path = self.get_resource_path(
            config, config.annotations.get(dataset_type)[imdb_file_index]
        )
        video_dir = self.get_resource_path(
            config, config.videos.get(dataset_type)[imdb_file_index]
        )
        classes_file = self.get_resource_path(config, config.classes_file)
        df = make_charades_df(
            csv_path=csv_path, video_dir=video_dir, classes_file=classes_file
        )

        precomputed_metadata = None
        pkl_path = os.path.join("charades", "defaults", f"metadata_{dataset_type}.pt")
        pkl_path = self.get_resource_path(config, pkl_path)

        if PathManager.exists(pkl_path):
            local_path = PathManager.get_local_path(pkl_path)
            with PathManager.open(local_path, "rb") as f:
                precomputed_metadata = torch.load(f)

        self.process_df(
            df,
            frames_per_clip=16,
            column_map={
                "labels": "action_labels",
                "video": "path",
                "text": "script",
                "id": "id",
            },
            num_workers=10,
            _precomputed_metadata=precomputed_metadata,
        )

        if not PathManager.exists(pkl_path):
            with PathManager.open(pkl_path, "wb") as f:
                torch.save(self.metadata, f)

    def get_resource_path(self, config, path):
        return os.path.join(config.data_dir, path)

    def process_df(
        self,
        df,
        frames_per_clip=16,
        column_map={},
        num_workers=1,
        _precomputed_metadata=None,
        **kwargs,
    ):
        self.labels = df[column_map.get("labels", "labels")].tolist()
        self.idx_to_class = sorted(
            list(set([item for sublist in self.labels for item in sublist]))
        )
        self.classes = self.idx_to_class
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.text_list = df[column_map.get("text", "text")].tolist()
        self.ids_list = df[column_map.get("id", "id")].tolist()

        video_list = df[column_map.get("video", "video")].tolist()
        self.video_clips = CharadesVideoClips(
            video_list,
            clip_length_in_frames=frames_per_clip,
            _precomputed_metadata=_precomputed_metadata,
            num_workers=num_workers,
        )

    @property
    def metadata(self):
        return self.video_clips.metadata

    def set_processors(self):
        if self.dataset_type == "train":
            self.video_processor = self.video_train_processor
        else:
            self.video_processor = self.video_test_processor

    def format_for_prediction(self, report):
        scores = torch.sigmoid(report.scores)
        binary_scores = scores > self.prediction_threshold
        predictions = []

        for idx, item_id in enumerate(report.id):
            item_id = byte_tensor_to_object(item_id)
            score = binary_scores[idx]
            labels = []
            score = score.nonzero(as_tuple=False)
            for item in score:
                labels.append(self.idx_to_class[item.item()])
            predictions.append({"id": item_id, "labels": labels})

        return predictions

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if len(self.video_clips) == 0:
            self.load_df()
        video, audio, info = self.video_clips.get_clip(idx)
        text = self.text_list[idx]
        actual_idx = self.ids_list[idx]
        label = [self.class_to_idx[class_name] for class_name in self.labels[idx]]
        one_hot_label = torch.zeros(len(self.class_to_idx))
        one_hot_label[label] = 1

        if self.video_processor is not None:
            video = self.video_processor(video)

        if self.audio_processor is not None:
            audio = self.audio_processor(audio)

        sample = Sample()
        sample.id = object_to_byte_tensor(actual_idx)
        sample.video = video
        sample.audio = audio
        sample.update(self.text_processor({"text": text}))
        sample.targets = one_hot_label
        return sample

    def show_clip(self, idx):
        from IPython.display import Audio, display, Image

        video, audio, text, one_hot = self[idx]
        # one hot to label index
        label = (
            torch.arange(one_hot.shape[0])[one_hot == 1].numpy().astype(int).tolist()
        )

        image_list = [PIL.Image.fromarray(frame.numpy()) for frame in video]

        audio_list = audio.numpy()

        path_to_gif = img2gif(image_list)

        display(
            "Labels: {}".format(str([self.classes[label_id] for label_id in label]))
        )
        display(Image(str(path_to_gif), format="png"))
        display(Audio(audio_list, rate=48000))
        display(text)
