import logging
import tempfile
from pathlib import Path

import pandas as pd
import torch
from torchvision.datasets.video_utils import VideoClips
from torchvision.io import read_video


logger = logging.getLogger(__name__)


def make_charades_df(csv_path, video_dir, classes_file):
    # load the csv
    logger.info(f"Reading from {csv_path}")
    df = pd.read_csv(csv_path)

    # transform the id to a pathname
    df["path"] = df["id"].map(lambda x: "{}/{}.mp4".format(video_dir, x))

    # parse action labels
    df["action_labels"] = df["actions"].map(
        lambda x: [label.split(" ")[0] for label in x.split(";")]
        if pd.notnull(x)
        else []
    )

    # load id to class map
    with open(classes_file, "r") as f:
        class_names = f.readlines()

    id2classname = {}
    for c in class_names:
        c_split = c.split(" ")
        assert len(c_split) > 1
        class_id = c_split[0]
        class_name = " ".join(c_split[1:]).strip("\n")
        id2classname[class_id] = class_name

    # transform label ids to names
    df["action_labels"] = df["action_labels"].map(
        lambda x: [id2classname[class_id] for class_id in x]
    )

    # filter only these videos that actually exist
    df_exists = df[df["path"].map(lambda x: Path(x).exists())]

    return df_exists


def img2gif(image_list, temporal_path="./tmp/"):
    tmp_file = Path(tempfile.NamedTemporaryFile(suffix=".gif").name).name
    tmp_file = Path(temporal_path) / tmp_file
    Path(tmp_file.parent).mkdir(exist_ok=True)
    print("Write to {}".format(tmp_file))
    with open(tmp_file, "wb") as tmp:
        image_list[0].save(
            tmp,
            format="GIF",
            append_images=image_list[1:],
            save_all=True,
            duration=3,
            loop=0,
        )
    return tmp_file


class CharadesVideoClips(VideoClips):
    @staticmethod
    def select_clips_from_video(video_pts, num_frames, fps, frame_rate):
        # this function replaces compute_clips_for_video from original Kinetics400
        # it yields one clip with evenly separated num_frames
        if fps is None:
            # if for some reason the video doesn't have fps
            # (because doesn't have a video stream) set the fps to 1.
            # The value doesn't matter, because video_pts is empty anyway
            fps = 1
        if frame_rate is None:
            frame_rate = fps

        idxs = torch.round(torch.linspace(0, len(video_pts) - 1, num_frames)).type(
            torch.LongTensor
        )

        video_pts = video_pts[idxs]

        return video_pts, idxs

    def compute_clips(self, num_frames, step, frame_rate=None):
        self.num_frames = num_frames
        self.step = step
        self.frame_rate = frame_rate
        self.clips = []
        self.resampling_idxs = []
        for video_pts, fps in zip(self.video_pts, self.video_fps):
            clips, idxs = self.select_clips_from_video(
                video_pts, num_frames, fps, frame_rate
            )
            self.clips.append(clips)
            self.resampling_idxs.append(idxs)

    def __len__(self):
        return self.num_clips()

    def num_videos(self):
        return len(self.video_paths)

    def num_clips(self):
        return len(self.clips)

    def get_clip(self, idx):
        """
        Gets a subclip from a list of videos.

        Arguments:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError(
                "Index {} out of range "
                "({} number of clips)".format(idx, self.num_clips())
            )
        video_path = self.video_paths[idx]
        clip_pts = self.clips[idx]

        from torchvision import get_video_backend

        backend = get_video_backend()

        if backend == "pyav":
            # check for invalid options
            if self._video_width != 0:
                raise ValueError("pyav backend doesn't support _video_width != 0")
            if self._video_height != 0:
                raise ValueError("pyav backend doesn't support _video_height != 0")
            if self._video_min_dimension != 0:
                raise ValueError(
                    "pyav backend doesn't support _video_min_dimension != 0"
                )
            if self._video_max_dimension != 0:
                raise ValueError(
                    "pyav backend doesn't support _video_max_dimension != 0"
                )
            if self._audio_samples != 0:
                raise ValueError("pyav backend doesn't support _audio_samples != 0")

        if backend == "pyav":
            assert len(clip_pts) > 0
            start_pts = clip_pts[0].item()
            end_pts = clip_pts[-1].item()
            video, audio, info = read_video(video_path, start_pts, end_pts)
        else:
            raise NotImplementedError(f"backend {backend} is not implemented.")

        resampling_idx = self.resampling_idxs[idx]
        if isinstance(resampling_idx, torch.Tensor):
            resampling_idx = resampling_idx - resampling_idx[0]
        video = video[resampling_idx]
        info["video_fps"] = self.frame_rate
        assert len(video) == self.num_frames, "{} x {}".format(
            video.shape, self.num_frames
        )
        return video, audio, info
