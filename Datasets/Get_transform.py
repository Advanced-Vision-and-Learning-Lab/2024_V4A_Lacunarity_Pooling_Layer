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
from torchvision import transforms
import torch.nn as nn
## Local external libraries

def get_transform(Network_parameters, input_size=224):
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    dataset_sampler = None

     
    if Dataset == 'MedMNIST':
        data_transforms = {
        'train': transforms.Compose([
        transforms.Resize((Network_parameters['resize_size'])),
        transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
            )
    ]),
        'test': transforms.Compose([
        transforms.Resize((Network_parameters['resize_size'])),
        transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
            )
    ]),
        }
        
    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset))
    
    return data_transforms

