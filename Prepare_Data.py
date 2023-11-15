# -*- coding: utf-8 -*-
"""
Create datasets and dataloaders for models
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import pdb
from sklearn.model_selection import train_test_split


import ssl
## PyTorch dependencies
import torch
## Local external libraries
from Datasets.Pytorch_Datasets import *
from Datasets.Get_transform import *
from Datasets import preprocess
from Datasets import loader

def collate_fn_train(data):
    data, labels, index = zip(*data)
    data = torch.stack(data)
    labels = torch.stack(labels)
    data = data_transforms["train"]({"image":data.float()})
    index = torch.Tensor(index)
    return data["image"].float(), labels.float(), index

def collate_fn_test(data):
    data, labels, index = zip(*data)
    data = torch.stack(data)
    labels = torch.stack(labels)
    data = data_transforms["test"]({"image":data.float()})
    index = torch.Tensor(index)
    return data["image"].float(), labels.float(), index

def Prepare_DataLoaders(Network_parameters, split):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    dataset_sampler = {}

    
    # Data augmentation and normalization for training
    # Just normalization and resize for test
    # Data transformations as described in:
    # http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf
    global data_transforms
    data_transforms = get_transform(Network_parameters, input_size=224)
       
    if Dataset == 'BloodMNIST':
        train_dataset = BloodMNIST(data_dir, split='train', transform = data_transforms['train'], target_transform=None)
        test_dataset = BloodMNIST(data_dir, split='test', transform = data_transforms['test'], target_transform=None)
        val_dataset = BloodMNIST(data_dir, split='val', transform = data_transforms['test'], target_transform=None)

    elif Dataset == 'PneumoniaMNIST':
        train_dataset = PneumoniaMNIST(data_dir, split='train', transform = data_transforms['train'], target_transform=None)
        test_dataset = PneumoniaMNIST(data_dir, split='test', transform = data_transforms['test'], target_transform=None)
        val_dataset = PneumoniaMNIST(data_dir, split='val', transform = data_transforms['test'], target_transform=None)

    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 

    dataset_sampler = {'train': None, 'test': None, 'val': None}
    #Create dataloaders
    image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    collate_fn = {'train': collate_fn_train, 'val': collate_fn_test, 'test': collate_fn_test}
    

    # Create training and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=Network_parameters['batch_size'][x],
                                                        shuffle=True,
                                                        )
                                                        for x in ['train', 'val','test']}
    

    return dataloaders_dict