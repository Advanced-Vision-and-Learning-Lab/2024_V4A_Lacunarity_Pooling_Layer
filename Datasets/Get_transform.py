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

# Data augmentation and normalization for training
# Just normalization and resize for test
# Data transformations as described in:
# http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf

def get_transform(Network_parameters, input_size=224):
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    
    if Dataset == 'LeavesTex':
                data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.3544, 0.4080, 0.1334], [0.0312, 0.0344, 0.0638])
            ]),
            'test': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.3544, 0.4080, 0.1334], [0.0312, 0.0344, 0.0638])
            ]),
        }
                
    elif Dataset == 'PlantVillage':
                data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.467, 0.489, 0.412], [0.177, 0.152, 0.194])
            ]),
            'test': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.467, 0.489, 0.412], [0.177, 0.152, 0.194])
            ]),
        }
        
    elif Dataset == 'DeepWeeds':
                data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.379, 0.39, 0.38], [0.224, 0.225, 0.223])
            ]),
            'test': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.379, 0.39, 0.38], [0.224, 0.225, 0.223])
            ]),
        }

    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset))
    
    return data_transforms
