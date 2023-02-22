# -*- coding: utf-8 -*-
# phoenixyli 李岩 @2020-04-02 14:33:59
import os
import numpy as np
from numpy.random import randint

import torch.utils.data as data
from PIL import Image

# afm add MGSampler
import json
with open('./MGSampler/features/S-S V1/train.json', 'r') as f:
    img_train = json.load(f)

with open('./MGSampler/features/S-S V1/val.json', 'r') as f:
    img_val = json.load(f)
# afm add end

class VideoRecord(object):
    """Store the basic information of the video

    _data[0]: the absolute path of the video frame folder
    _data[1]: the frame number
    _data[2]: the label of the video
    """

    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    """The torch dataset for the video data.

    :param list_file: the list file is utilized to specify the data sources.
    Each line of the list file contains a tuple of extracted video frame folder path (absolute path),
    video frame number, and video groundtruth class. An example line looks like:
    /data/xxx/xxx/Dataset/something-somthing-v1/100218 42 134
    """

    def __init__(
            self, root_path, list_file, num_segments=8, new_length=1, modality='RGB',
            image_tmpl='img_{:05d}.jpg', transform=None, random_shift=True,
            test_mode=False, remove_missing=False, multi_clip_test=False,
            dense_sample=False, is_train=None, temp_transform=None):
        
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.multi_clip_test = multi_clip_test
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.is_train = is_train # for RandAugment
        self.temp_transform = temp_transform # for snip sampling
        self.new_length = 3 # for snip sampling
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(directory, self.image_tmpl.format(1))).convert('RGB')]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """Random Sampling from each video segment

        :param record: VideoRecord
        :return: list
        """

        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration)\
                          + randint(average_duration, size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        """Sampling for validation set

        Sample the middle frame from each video segment
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        file_name = self.image_tmpl.format(1)
        full_path = os.path.join(self.root_path, record.path, file_name)
        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(record.path, file_name)
        
        
        if not self.test_mode:  # training or validation set
            if self.random_shift:  # training set
                segment_indices = self._sample_indices(record)
            else:  # validation set
                segment_indices = self._get_val_indices(record)
        else:  # test set
            # for mulitple clip test, use random sampling;
            # for single clip test, use middle sampling
            if self.multi_clip_test:
                segment_indices = self._sample_indices(record)
            else:
              segment_indices = self._get_test_indices(record)
        
        '''
        # afm add MGSampler
        if self.is_train:
            img_diff = img_train[record.path]
            MGSampler = SampleFrames(clip_len=1, frame_interval=1, num_clips=8, temporal_jitter=False, 
                                      twice_sample=False, out_of_bound_opt='loop', test_mode=False)
            segment_indices = MGSampler(record,img_diff)
        else:
            img_diff = img_val[record.path]
            MGSampler = SampleFrames(clip_len=1, frame_interval=1, num_clips=8, temporal_jitter=False, 
                                      twice_sample=False, out_of_bound_opt='loop', test_mode=True)
            segment_indices = MGSampler(record,img_diff)
        # afm add end
        '''
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        idx_list = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                idx_list.append(p)
                if p < record.num_frames:
                    p += 1

        process_idx_list = self.temp_transform(idx_list)
        for p in process_idx_list:
            seg_imgs = self._load_image(os.path.join(self.root_path, record.path), p)
            images.extend(seg_imgs)
        
        
        if self.is_train:
            # afm add RandAugment
            from ops.augmentations import RandAugment
            augment_images = []
            randaugment_factory = RandAugment(1,1)
            for image in images:
                image = randaugment_factory(image)
                augment_images.append(image)
            process_data = self.transform(augment_images)
        else:
            process_data = self.transform(images)

            
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


import warnings
import random
class SampleFrames(object):
    """Sample frames from the video.
    Required keys are "filename", "total_frames", "start_index" , added or
    modified keys are "frame_inds", "frame_interval" and "num_clips".
    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.
        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.
        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, results, img_diff):

        def find_nearest(array, value):
            array = np.asarray(array)
            try:
                idx = (np.abs(array - value)).argmin()
                return int(idx + 1)
            except(ValueError):
                print(results['filename'])

        #diff_score = results['img_diff']
        diff_score = img_diff
        diff_score = np.power(diff_score, 0.5)
        sum_num = np.sum(diff_score)
        diff_score = diff_score / sum_num

        count = 0
        pic_diff = list()
        for i in range(len(diff_score)):
            count = count + diff_score[i]
            pic_diff.append(count)

        choose_index = list()
        
        if self.test_mode:
            choose_index.append(find_nearest(pic_diff, 1 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 16 + 1 / 8))
            choose_index.append(find_nearest(pic_diff, 1 / 16 + 2 / 8))
            choose_index.append(find_nearest(pic_diff, 1 / 16 + 3 / 8))
            choose_index.append(find_nearest(pic_diff, 1 / 16 + 4 / 8))
            choose_index.append(find_nearest(pic_diff, 1 / 16 + 5 / 8))
            choose_index.append(find_nearest(pic_diff, 1 / 16 + 6 / 8))
            choose_index.append(find_nearest(pic_diff, 1 / 16 + 7 / 8))
        else:
            choose_index.append(find_nearest(pic_diff, random.uniform(0, 1 / 8)))
            choose_index.append(find_nearest(pic_diff, random.uniform(1 / 8, 2 / 8)))
            choose_index.append(find_nearest(pic_diff, random.uniform(2 / 8, 3 / 8)))
            choose_index.append(find_nearest(pic_diff, random.uniform(3 / 8, 4 / 8)))
            choose_index.append(find_nearest(pic_diff, random.uniform(4 / 8, 5 / 8)))
            choose_index.append(find_nearest(pic_diff, random.uniform(5 / 8, 6 / 8)))
            choose_index.append(find_nearest(pic_diff, random.uniform(6 / 8, 7 / 8)))
            choose_index.append(find_nearest(pic_diff, random.uniform(7 / 8, 8 / 8)))

        return choose_index

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


if __name__ == "__main__":
    # test dataset
    test_train_list ='/data1/phoenixyli/DeepLearning/something-something-v1/TrainTestlist/val_videofolder_new.txt'
    test_num_segments = 8
    data_length = 1
    test_modality = 'RGB'
    prefix = '{:05d}.jpg'
    train_dataset = TSNDataSet(
        test_train_list, num_segments=test_num_segments,
        new_length=data_length, modality=test_modality,
        image_tmpl=prefix, multi_clip_test=False, dense_sample=False)
    data, label = train_dataset.__getitem__(10)

