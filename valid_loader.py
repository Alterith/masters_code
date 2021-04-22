# torch imports
import torch
from torch.utils.data import Dataset

# generic imports
import os
import numpy as np
import random
import pandas as pd
from decord import VideoReader
import cv2
import pickle
import linecache as lc

# create data loader

class video_dataset(Dataset):
    def __init__(self, data_dir, split, temporal_depth, patch_width, patch_height, dataset_name):
        print(data_dir)
        # list of classes
        if split == "test":
            self.classes = os.listdir(os.path.join(data_dir, "val"))
        else:
            self.classes = os.listdir(os.path.join(data_dir, split))
        #print(self.classes)
        # reduce kinetics_600 to 400
        if dataset_name == "kinetics_368":
            kinetics_400_list = pd.read_csv(data_dir+"/kinetics_368_classes.csv", header=None)
            kinetics_400_list = kinetics_400_list.values.tolist()
            kinetics_400_list = [item for sublist in kinetics_400_list for item in sublist]
            #kinetics_400_list = [kinetics_400_list[0], kinetics_400_list[1]]
            #print(kinetics_400_list)
            self.classes = [cls for cls in self.classes if cls in kinetics_400_list]
            # testing code
            #self.classes = [self.classes[0], self.classes[1]]
        # list of each class directory
        if split == "test":
            self.classes_dir = [os.path.join(os.path.join(data_dir, "val"), f) for f in self.classes]
        else:
            self.classes_dir = [os.path.join(os.path.join(data_dir, split), f) for f in self.classes]

        # list of the video file directories in each class folder
        self.data_list_by_dir = []
        self.idx_per_file = []
        # list of lists each containing the video drectories per class
        class_order_file = 'kinectics_368_order.txt'
        for i, f in enumerate(self.classes_dir):
            f = f.split("/")
            f[-1] = lc.getline(class_order_file, i+1)[:-1]
            f = "/".join(f)
            tmp = os.listdir(os.path.join(f, ""))
            #print(len(tmp))
            if split == "val":
                with open(data_dir+"/test_vids.txt") as test_vid_file:
                    test_vids = test_vid_file.read().splitlines()
                tmp = [v for v in tmp if v not in test_vids]
            elif split == "test":
                with open(data_dir+"/test_vids.txt") as test_vid_file:
                    test_vids = test_vid_file.read().splitlines()
                tmp = [v for v in tmp if v in test_vids]
            #print(len(tmp))
            #quit()
            self.data_list_by_dir.append([os.path.join(os.path.join(f, ""), g) for g in tmp])
            self.idx_per_file.append(np.empty(len(self.data_list_by_dir[i])))
            self.idx_per_file[i].fill(i)
            print(i, f)


        #flatten indicies and directories
        self.idx_per_file = [item for sublist in self.idx_per_file for item in sublist]
        self.idx_per_file = [np.int64(idx) for idx in self.idx_per_file]
        self.flattened_data_dir = [item for sublist in self.data_list_by_dir for item in sublist]
        # print(len(self.flattened_data_dir))
        # train, test, val
        self.split = split

        # number of consecutive frames
        self.temporal_depth = temporal_depth

        # dimension of patch selected
        self.patch_width = patch_width
        self.patch_height = patch_height
        #print(self.classes)
        # with open('kraken/kinetics_400_order_val.data', 'rb') as f:
        #     self.flattened_data_dir = pickle.load(f)

        # with open('kraken/kinetics_400_order_classes_val.data', 'rb') as f:
        #     self.idx_per_file= pickle.load(f)


    #data augnemtation transforms
    def transform(self, vid, split):

        total_frames = int(len(vid))

        #print(total_frames)

        vid_width = vid[0].shape[1]

        vid_height = vid[0].shape[0]

        clips = []


        # frame iterator / stride index
        stride = 0
        stride_index = self.temporal_depth

        # obtrain the temporal depth number of consecutive frames

        while(stride*stride_index + self.temporal_depth < total_frames):

            imgs = []
            start_frame = stride*stride_index

            for i in range(start_frame, start_frame + self.temporal_depth):

                frame = vid[i]
                frame = frame.asnumpy()
                frame = frame.astype(np.float32)
                #frame = np.asarray(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (112, 112), interpolation = cv2.INTER_LINEAR) # remove this or move it up

                cv2.normalize(frame, frame, 0, 1, cv2.NORM_MINMAX)
                imgs.append(frame)

            stride = stride + 1
            clips.append(imgs)

        clips = np.asarray(clips)
        clips = clips.astype(np.float32)
        clips = np.moveaxis(clips, 4, 1)
        clips = torch.from_numpy(clips)

        return clips


    def __len__(self):

        return len(self.flattened_data_dir)


    def __getitem__(self, idx):

        result = False
        vid = None
        cls = None
        #idx = None #random.randint(0,400)
        # deal with corrupted videos in list
        #print(self.flattened_data_dir[idx])
        while not result:
            try:
                #vid = pims.PyAVVideoReader(self.flattened_data_dir[idx])
                vid = VideoReader(self.flattened_data_dir[idx])
                cls = self.idx_per_file[idx]
                test_frame = vid[1]
                if(int(len(vid))>self.temporal_depth):
                    result = True
                else:
                    idx = random.randint(0, len(self.flattened_data_dir)-1)
            except:
                idx = random.randint(0, len(self.flattened_data_dir)-1)

        frames = self.transform(vid, self.split)
        #del rand_vid, vid
        #del vid
        #print(frames.shape)
        return frames, cls
