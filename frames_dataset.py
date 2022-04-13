import glob
import json
import os
import struct
from io import BytesIO

import cv2
import numpy as np
from cv2 import COLOR_BGR2RGB, cvtColor, imread, resize
from imageio import mimread
from PIL import Image
from skimage import img_as_float32, io
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from augmentation import AllAugmentationTransform


def decode(file_name, file_seek):
    with open(file_name, 'rb') as stream:
        stream.seek(file_seek)

        l = struct.unpack('I', stream.read(4))[0]
        name = stream.read(l)

        l = struct.unpack('I', stream.read(4))[0]
        cont = stream.read(l)

        return (name, cont)



def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class TalkingHeadFramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(
        self,
        root_dir,
        n_videos_per_bin=512,
        frame_shape=(256, 256, 3),
        id_sampling=True,
        is_train=True,
        augmentation_params={
            "flip_param": {"horizontal_flip": True, "time_flip": True},
            "jitter_param": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.1},
        },
    ):
        self.root_dir = root_dir
        assert frame_shape[0] == frame_shape[1]
        self.frame_shape = tuple(frame_shape)
        self.id_sampling = id_sampling
        self.is_train = is_train

        self.n_videos_per_bin = n_videos_per_bin

        self.split = 'train' if self.is_train else 'val'
        # self.split = 'val'
        self.bin_dir = os.path.join(self.root_dir, f'bin/b{n_videos_per_bin}')
        self.ant_dir = os.path.join(self.bin_dir, 'annotations')
        with open(f'{self.ant_dir}/{self.split}_valid.json', 'r') as stream:
            self.data = json.load(stream)
        self.videos = self.data['videos']
        self.transform = AllAugmentationTransform(**augmentation_params) if self.is_train else None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_name = self.videos[idx]
        num_frames = len(self.data[video_name])
        video_array = []

        if self.is_train:
            frame_ids = np.sort(np.random.choice(num_frames, replace=True, size=2))
        else:
            frame_ids = [i for i in range(num_frames)]
        for frame_id in frame_ids:
            record = self.data[video_name][frame_id]
            bin_file_name = record['bin_file_name']
            bin_file_path = os.path.join(self.bin_dir, self.split, bin_file_name)
            file_seek = record['bin_file_seek']
            _, cont = decode(bin_file_path, file_seek)
            image = Image.open(BytesIO(cont))
            image = np.array(image)
            video_array.append(image)

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype="float32")
            driving = np.array(video_array[1], dtype="float32")

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype="float32")
            out['video'] = video.transpose((3, 0, 1, 2))
        out['name'] = video_name

        return out

class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
