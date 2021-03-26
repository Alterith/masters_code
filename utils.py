import numpy as np

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import Sampler
import torch.nn.init as init

# data loader imports
from data_loader_classifier import video_dataset

from data_loader_captions import caption_dataset

def calc_proposal_weights(target_tensor):
    """
    calculates the binary weights for a given tensor for each dimension along axis 0
    Args:
        target_tensor: (torch.Tensor) is the tensor for which we calculate the binary weights [batch, num_points]
    Returns:
        pos_weights: (torch.Tensor) contains the pos_class weights for bcewithlogitsloss [1, num_points]
    """
    pos_weights = None
    gt_unique_bins = None
    gt_cat_counts = None
    for i in range(0, target_tensor.shape[1]):
        gt_unique, gt_counts = torch.unique(target_tensor[:,i], return_counts=True)

        # if 1 isnt in a cluster append it with 0 count to the counts and bins
        if not 1 in gt_unique:
            gt_unique = torch.cat((gt_unique, torch.Tensor([1])), dim = 0)
            gt_counts = torch.cat((gt_counts, torch.Tensor([0])), dim = 0)

        # if 0 isnt in a cluster append it with 0 count to the counts and bins
        if not 0 in gt_unique:
            gt_unique = torch.cat((torch.Tensor([0]), gt_unique), dim = 0)
            gt_counts = torch.cat((torch.Tensor([0]), gt_counts), dim = 0)

        if i == 0:
            gt_unique_bins = gt_unique
            gt_cat_counts = gt_counts.unsqueeze(0)
        else:
            gt_counts = gt_counts.unsqueeze(0)
            gt_cat_counts = torch.cat((gt_cat_counts, gt_counts), dim=0)

    # add epsilon for non-zero division
    gt_cat_counts = gt_cat_counts.float()
    gt_cat_counts[:,1] += 1e-3
    # calculate positive weights neg_total/pos_total
    pos_weights = torch.div(gt_cat_counts[:,0], gt_cat_counts[:,1])

    return pos_weights.cpu().detach().numpy()

def calc_caption_weights(target_tensor, target_mask, end_id, options):
    """
    calculates the weights for a given tensor for input into crossentropy loss
    Args:
        target_tensor: (torch.Tensor) is the tensor for which we calculate the weights [1,1,num_features, 30]
        target_mask: (torch.Tensor) is the tensor for which we calculate target tensor mask [1,1,num_features, 30]
        end_id: (torch.Tensor) is the tensor which contains the end id's for captions
        options: (dict) are the default options used
    Returns:
        weights: (torch.Tensor) contains the class weights for crossentropy loss [1, len(vocab)]
    """
    # create torch tensor with len of vocab and fill with initial value
    weights = torch.empty(options['vocab_size']).fill_(0.25)
    unique_words = None
    target_masked = []
    for i in range(0, end_id.shape[0]):

        target_masked.append(target_tensor[0][0][end_id][i][torch.nonzero(target_mask[0][0][end_id][i]).view(-1)].tolist())

    # get the unique words after mask is applied
    flattened_target_masked = [item for sublist in target_masked for item in sublist]
    flattened_target_masked = np.asarray(flattened_target_masked)
    
    unique_words, unique_counts = torch.unique(flattened_target_masked, return_counts=True)
    # this is just a weighted mean calculation per video
    unique_weights = [float(sum(unique_counts))/float(len(unique_counts) * cnt) for cnt in unique_counts]

    # assign weights to the unique words
    for idx in unique_words:
        weights[idx] = unique_weights[idx]

    return weights.cpu().detach().numpy()

# taken from https://discuss.pytorch.org/t/strange-behavior-with-sgd-momentum-training/7442 written by @smth
class WithReplacementRandomSampler(Sampler):
    """Samples elements randomly, with replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # generate samples of `len(data_source)` that are of value from `0` to `len(data_source)-1`
        samples = torch.LongTensor(len(self.data_source))
        samples.random_(0, len(self.data_source))
        return iter(samples)

    def __len__(self):
        return self.num_samples


class WithoutReplacementRandomSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        # generate samples of `len(data_source)` that are of value from `0` to `len(data_source)-1`
        samples = torch.randperm(len(self.data_source))[:self.num_samples]
        return iter(samples)

    def __len__(self):
        return len(self.data_source)

def fetch_dataloader(types, opt, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    datasets = {}
    for split in types:
        
        dataset = None
        if params['data_type'] == 'video':
            dataset = video_dataset(opt, split)
        else:
            dataset = caption_dataset(opt, split)
        if split == 'train':
            if opt['train_use_sampler'] == True:
                vid_dataset_sampler = RandomSampler(dataset, replacement=True, num_samples=params['num_vids_per_epoch'])
            else:
                # this is prefered over torch shuffle as we encountered issues where classes were grouped together leading to problems during training
                vid_dataset_sampler = WithReplacementRandomSampler(dataset)

            dl = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False,
                            num_workers=params['num_workers'], sampler = vid_dataset_sampler)
        else:
            if opt['val_use_sampler'] == True:
                vid_dataset_sampler = WithoutReplacementRandomSampler(dataset, opt['metric_eval_num'])
            else:
                vid_dataset_sampler = None
            dl = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False,
                            num_workers=params['num_workers'], sampler = vid_dataset_sampler)

        dataloaders[split] = dl
        datasets[split] = dataset
    return dataloaders, datasets

# taken from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5 with minor additions
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        init.torch.nn.init.uniform_(m.weight.data, -0.08, 0.08)

# taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
## Updated by Dhruv Bhugwan
'''
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # print(correct[:k].reshape(-1).float().shape)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        #res = np.asarray(res)
        return res
