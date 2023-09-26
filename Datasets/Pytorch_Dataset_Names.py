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

    if Dataset == 'UCMerced':
        dataset = UCMerced_Index(data_dir, split='train',download=True)

    elif Dataset == 'EuroSAT':
        dataset = Eurosat_Index(data_dir, split='train',download=True)  

    elif Dataset == 'Resisc45':
        dataset = Resisc45_Index(data_dir, split='train',download=True)
    
    elif Dataset == 'Eurosat_MSI':
        dataset = Eurosat_MSI(data_dir, split='train',download=True)

    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 

     
    #Return class names
    if dataset:
        class_names = dataset.classes
    
    return class_names