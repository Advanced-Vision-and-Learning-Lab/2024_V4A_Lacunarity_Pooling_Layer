# -*- coding: utf-8 -*-
"""
Create datasets and dataloaders for models
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import pdb

## PyTorch dependencies
import torch
from torchvision import transforms
import torch.nn as nn


def get_transform(Network_parameters, input_size=224):
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    
    if Dataset == 'BloodMNIST' or Dataset == 'PneumoniaMNIST':
        mean = [0]
        std = [1]
        data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(Network_parameters['resize_size']),
            transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
        ]),
        'test':  transforms.Compose([
            transforms.Resize(Network_parameters['resize_size']),
            transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
        ]),
    }
        
    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset))
    
    return data_transforms, mean, std

