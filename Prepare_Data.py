# -*- coding: utf-8 -*-
"""
Create datasets and dataloaders for models
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import pdb
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from pytorch_metric_learning import samplers

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

def Prepare_DataLoaders(Network_parameters, split,input_size=224, view_results = True):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']

    
    # Data augmentation and normalization for training
    # Just normalization and resize for test
    # Data transformations as described in:
    # http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf
    global data_transforms
    data_transforms = get_transform(Network_parameters, input_size=224)
    dataset_sampler = None

    if Dataset == 'FashionMNIST':
        train_dataset = FashionMNIST_Index(data_dir,train=True,transform=data_transforms['train'],
                                        download=True)
        y = train_dataset.targets
        indices = np.arange(len(y))
        indices = torch.as_tensor(indices)
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                            test_size=0.1, 
                                                            stratify=y, random_state=42)
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        dataset_sampler = {'train': train_sampler, 'val': val_sampler, 'test': None}
        
        test_dataset = FashionMNIST_Index(data_dir,train=False,transform=data_transforms['test'],
                                        download=True)
            
       
    elif Dataset == 'BloodMNIST':
        train_dataset = BloodMNIST(data_dir, split='train', transform = data_transforms['train'], target_transform=None)
        test_dataset = BloodMNIST(data_dir, split='test', transform = data_transforms['test'], target_transform=None)
        val_dataset = BloodMNIST(data_dir, split='val', transform = data_transforms['test'], target_transform=None)

    elif Dataset == 'PneumoniaMNIST':
        train_dataset = PneumoniaMNIST(data_dir, split='train', transform = data_transforms['train'], target_transform=None)
        test_dataset = PneumoniaMNIST(data_dir, split='test', transform = data_transforms['test'], target_transform=None)
        val_dataset = PneumoniaMNIST(data_dir, split='val', transform = data_transforms['test'], target_transform=None)

    elif Dataset == 'OrganMNISTCoronal':
        train_dataset = OrganMNISTCoronal(data_dir, split='train', transform = data_transforms['train'], target_transform=None)
        test_dataset = OrganMNISTCoronal(data_dir, split='test', transform = data_transforms['test'], target_transform=None)
        val_dataset = OrganMNISTCoronal(data_dir, split='val', transform = data_transforms['test'], target_transform=None)

    elif Dataset == 'PlantLeaf':
        train_dataset = PlantLeaf(data_dir, split='train', transform = data_transforms['train'])
        test_dataset = PlantLeaf(data_dir, split='test', transform = data_transforms['test'])
        val_dataset = PlantLeaf(data_dir, split='val', transform = data_transforms['test'])

    elif Dataset == 'UCMerced': #See people also use .5, .5 for normalization
        train_dataset = UCMerced_Index(root = data_dir,split='train', transform = data_transforms['train'], download=True)
        test_dataset = UCMerced_Index(data_dir,split='test', transform = data_transforms['test'], download=True)
        val_dataset = UCMerced_Index(data_dir,split='val', transform = data_transforms['test'], download=True)
        

    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 
    
    if Dataset == "FashionMNIST" or Dataset == "UCMerced":
        labels = test_dataset.targets
        classes = test_dataset.classes
        #m is the number of samples taken from each class
        m = 10
        #In our paper, batch_size for:
            #UCMerced - 210
            #EuroSAT - 100
            #MSTAR - 40
        batch_size = m*len(classes)
        sampler = samplers.MPerClassSampler(labels, m, batch_size, length_before_new_iter=100000)
        #retain sampler = None for 'train' and 'val' data splits
        dataset_sampler = {'train': None, 'test': sampler, 'val': None}
        Network_parameters["batch_size"]["test"] = batch_size

    else:
        dataset_sampler = {'train': None, 'test': None, 'val': None}

    #Collate function is used only for EuroSAT and MSTAR
    #Compatible input size for Kornia augmentation
    if Dataset == "UCMerced":
    # Create training and test dataloaders
        image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=Network_parameters['batch_size'][x],
                                                        num_workers=Network_parameters['num_workers'],
                                                        pin_memory=Network_parameters['pin_memory'],
                                                        sampler = dataset_sampler[x])
                                                        for x in ['train', 'val','test']}

    # Create training and test dataloaders
    #for FashionMNIST dataset
    elif Dataset == "FashionMNIST":
        image_datasets = {'train': train_dataset, 'val': train_dataset, 'test': test_dataset}
        # Create training and test dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                           batch_size=Network_parameters['batch_size'][x],
                                                           sampler=dataset_sampler[x],
                                                           num_workers=Network_parameters['num_workers'],
                                                           pin_memory=Network_parameters['pin_memory'])
                            for x in ['train', 'val', 'test']}
    else:
        image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=Network_parameters['batch_size'][x],
                                                        shuffle=True,
                                                        )
                                                        for x in ['train', 'val','test']}
    

    return dataloaders_dict