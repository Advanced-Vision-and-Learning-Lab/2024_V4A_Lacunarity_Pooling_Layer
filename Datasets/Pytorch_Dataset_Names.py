# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:59:50 2023
Return names of classes in each dataset
@author: jpeeples
"""

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision import datasets
from Datasets.Pytorch_Datasets import *



def Get_Class_Names(Dataset,data_dir):

    if Dataset == 'BloodMNIST':
        dataset = BloodMNIST(data_dir, split='train',download=True)

    elif Dataset == 'PneumoniaMNIST':
        dataset = PneumoniaMNIST(data_dir, split='train',download=True)

    elif Dataset == 'OrganMNISTCoronal':
        dataset = OrganMNISTCoronal(data_dir, split='train',download=True)
    
    elif Dataset == 'FashionMNIST':
        dataset = FashionMNIST_Index(data_dir, train=True,download=True)

    elif Dataset == "PlantLeaf":
        dataset = PlantLeaf(data_dir, split = "train")

    elif Dataset == "UCMerced":
        dataset = UCMerced_Index(data_dir, split='train',download=True)

    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 

     
    #Return class names
    if dataset:
        class_names = dataset.classes
    
    return class_names