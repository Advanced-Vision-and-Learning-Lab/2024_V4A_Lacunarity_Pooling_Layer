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


## Local external libraries
from Datasets import preprocess

def get_transform(Network_parameters, input_size=224):
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    
    if Dataset == 'BloodMNIST' or Dataset == 'PneumoniaMNIST' or Dataset == 'OrganMNISTCoronal':
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
    elif Dataset == 'FashionMNIST':
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5]),
            ]),
            'test':  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5]),
            ]),
    }
        
    elif Dataset == 'PRMI':
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.4077349007129669, 0.3747502267360687, 0.34903043508529663], [0.4077349007129669, 0.3747502267360687, 0.34903043508529663])
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.4077349007129669, 0.3747502267360687, 0.34903043508529663], [0.4077349007129669, 0.3747502267360687, 0.34903043508529663])
            ]),
        }


    elif Dataset == "PlantLeaf":
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(size = (200, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5]),

            ]),
            'test':  transforms.Compose([
                transforms.Resize(size = (200, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5]),
            ]),
    } 

    elif Dataset == 'UCMerced':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(Network_parameters['resize_size']),
            transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(Network_parameters['center_size']),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    elif Dataset == "Synthetic_Gray" or Dataset == "Synthetic_RGB":
        if 'Grayscale' in Dataset:
            data_transforms = {
                'val': transforms.Compose([
                    transforms.Resize(size=(200, 200)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                ]),
            }
        
        else:
            data_transforms = {
                'val': transforms.Compose([
                    transforms.Resize(size=(200, 200)),
                    transforms.ToTensor(),
                ]),
            }
        
    elif Dataset == "Kth_Tips" or Dataset == "GTOS-mobile":  
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
    elif Dataset == 'LeavesTex':
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
