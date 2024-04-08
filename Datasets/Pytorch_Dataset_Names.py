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

    if Dataset == 'LeavesTex':
        dataset = LeavesTex1200(data_dir)

    elif Dataset == 'PlantVillage':
        dataset = PlantVillage(data_dir)
    
    elif Dataset == 'DeepWeeds':
        dataset = DeepWeeds(data_dir)

    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 

     
    #Return class names
    if dataset:
        class_names = dataset.classes
    
    return class_names