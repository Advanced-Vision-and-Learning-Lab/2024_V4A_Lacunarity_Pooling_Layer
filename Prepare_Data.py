# -*- coding: utf-8 -*-
"""
Create datasets and dataloaders for models
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import itertools
import pdb
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

import ssl
## PyTorch dependencies
import torch
from torchvision import transforms
import torch.nn as nn

## Local external libraries
from Datasets.Pytorch_Datasets import *
from Datasets.Get_transform import *

def collate_fn_train(data):
    index, data, labels = zip(*data)
    data = torch.stack(data)
    labels = torch.stack(labels)
    data = data_transforms["train"]({"image":data.float()})
    index = torch.Tensor(index)
    return data["image"].float(), labels.float(), index

def collate_fn_test(data):
    index, data, labels = zip(*data)
    data = torch.stack(data)
    labels = torch.stack(labels)
    data = data_transforms["test"]({"image":data.float()})
    index = torch.Tensor(index)
    return data["image"].float(), labels.float(), index

def Prepare_DataLoaders(Network_parameters, split,input_size=224):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    dataset_sampler = None

    # Data augmentation and normalization for training
    # Just normalization and resize for test
    # Data transformations as described in:
    # http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf
    global data_transforms
    data_transforms = get_transform(Network_parameters, input_size=input_size)

   
       
    if Dataset == 'MedMNIST': #See people also use .5, .5 for normalization
        train_dataset = MedMNIST(data_dir, split='train', transform = data_transforms['train'], target_transform=None)
        test_dataset = MedMNIST(data_dir, split='test', transform = data_transforms['test'], target_transform=None)
        val_dataset = MedMNIST(data_dir, split='val', transform = data_transforms['test'], target_transform=None)


    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 

    #Create dataloaders
    image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    collate_fn = {'train': collate_fn_train, 'val': collate_fn_test, 'test': collate_fn_test}
    # Create training and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=Network_parameters['batch_size'][x],
                                                        shuffle=True,
                                                        num_workers=Network_parameters['num_workers'],
                                                        pin_memory=Network_parameters['pin_memory'],
                                                        )
                                                        for x in ['train', 'val','test']}

    return dataloaders_dict