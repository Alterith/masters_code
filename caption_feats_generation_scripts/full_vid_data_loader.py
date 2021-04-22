import h5py

# torch imports
import torch
from torch.utils.data import Dataset

# generic imports
import os
import sys
import numpy as np
import random
import pandas as pd
import cv2

from decord import VideoReader
from decord import cpu, gpu

from matplotlib import pyplot as plt
import gc

# create data loader

class video_dataset(Dataset):
    def __init__(self, data_dir, split, temporal_depth, patch_width, patch_height, dataset_name, stride=None, stride_idx=None):
        print(data_dir)
        # list of classes
        self.vids = os.listdir(os.path.join(data_dir, split))

        # list of the video file directories in each class folder
        self.flattened_data_dir = [os.path.join(os.path.join(data_dir, split),v) for v in self.vids]
        if stride is not None and stride_idx is not None:
            try:
                if stride*(stride_idx+1) <= len(self.flattened_data_dir):
                    self.flattened_data_dir = self.flattened_data_dir[stride*stride_idx:stride*(stride_idx+1)]
                else:
                    self.flattened_data_dir = self.flattened_data_dir[stride*stride_idx:]
            except Exception as e:
                print("Dataloader out of range")
                quit()

        # train, test, val
        self.split = split

        # number of consecutive frames
        self.temporal_depth = temporal_depth

        # dimension of patch selected
        self.patch_width = patch_width
        self.patch_height = patch_height



    #data augnemtation transforms
    def transform(self, vid, split):

        total_frames = int(len(vid))

        print(total_frames)

        if total_frames > 7200:
            return torch.zeros(901, 1, 1, 1)

        vid_width = vid[0].shape[1]

        vid_height = vid[0].shape[0]

        start_frame = random.randint(0, (total_frames - self.temporal_depth - 1))
        patch_start_width = random.randint(0, 171 - self.patch_width - 1)
        patch_start_height = random.randint(0, 128 - self.patch_height - 1)
        clips = []

        # the prob of flipping a video
        flip_prob = random.random()

        # frame iterator / stride index
        stride = 0
        stride_index = self.temporal_depth

        # obtrain the temporal depth number of consecutive frames

        inter_method_idx = 0 #random.randint(0,4)
        inter_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

        while(stride*stride_index + self.temporal_depth < total_frames):

            imgs = []
            start_frame = stride*stride_index

            for i in range(start_frame, start_frame + self.temporal_depth):

                frame = vid[i]

                
                # frame = frame.astype()
                frame = frame.asnumpy()
                frame = frame.astype(np.float32)
                frame = np.asarray(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print(frame)
                # plt.imshow(frame)
                # plt.show()
                
                # quit()
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (112, 112), interpolation = inter_methods[inter_method_idx]) # remove this or move it up

                cv2.normalize(frame, frame, 0, 1, cv2.NORM_MINMAX)
                imgs.append(frame)

            stride = stride + 1
            clips.append(imgs)

        clips = np.asarray(clips, dtype=np.float32)
        clips = clips.astype(np.float32)
        clips = np.moveaxis(clips, 4, 1)
        clips = torch.from_numpy(clips)

        return clips


    def __len__(self):

        return len(self.flattened_data_dir)


    def __getitem__(self, idx):

        if idx < 0:
            return torch.zeros(1, 1, 1, 1), self.flattened_data_dir[idx]

        result = False
        vid = None
        # idx = 3456
        # deal with corrupted videos in list or videos which are just too long for us to process
        while not result:
            try:
                vid = VideoReader(self.flattened_data_dir[idx])
                if(int(len(vid))>self.temporal_depth):
                    result = True
                else:
                    #idx = random.randint(0, len(self.flattened_data_dir)-1)
                    del vid
                    gc.collect()
                    return torch.zeros(901, 1, 1, 1), -1
            except:
                #idx = random.randint(0, len(self.flattened_data_dir)-1)
                del vid
                gc.collect()
                return torch.zeros(901, 1, 1, 1), -1

        frames = self.transform(vid, self.split)

        # vid.close()
        
        del vid
        gc.collect()
        return frames, self.flattened_data_dir[idx]
