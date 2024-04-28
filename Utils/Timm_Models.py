# -*- coding: utf-8 -*-
"""
Functions to train/test model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from torchvision import models
import pdb
import os
import torch.nn.functional as F
from Utils.Get_Pooling import get_pooling
import timm

class densenet161_lacunarity(nn.Module):
    def __init__(self, num_ftrs, num_classes, Params, pooling_layer="lacunarity", agg_func="global"):

        super(densenet161_lacunarity, self).__init__()
        self.model_dense = timm.create_model('densenet161', pretrained=True, num_classes=0, global_pool='')
        model_name = Params['Model_name']
        self.feature_extraction = Params["feature_extraction"]
        self.set_parameter_requires_grad()
        
        self.avgpool = get_pooling(model_name, Params)
        
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.agg_func = agg_func
        


    def forward(self,x):
        features = self.model_dense(x)
        out = F.relu(features, inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    
    def set_parameter_requires_grad(self):
        if self.feature_extraction:
            for param in self.model_dense.parameters():
                param.requires_grad = False



class resnet18_lacunarity(nn.Module):
    def __init__(self, num_ftrs, num_classes, Params, pooling_layer="lacunarity", agg_func="global"):

        super(resnet18_lacunarity, self).__init__()
        self.model_dense = timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool='')
        model_name = Params['Model_name']
        self.feature_extraction = Params["feature_extraction"]
        self.set_parameter_requires_grad()
        
        self.avgpool = get_pooling(model_name, Params)
        
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.agg_func = agg_func
        


    def forward(self,x):
        features = self.model_dense(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    
    def set_parameter_requires_grad(self):
        if self.feature_extraction:
            for param in self.model_dense.parameters():
                param.requires_grad = False


class convnext_lacunarity(nn.Module):
    def __init__(self, num_ftrs, num_classes, Params, pooling_layer="lacunarity", agg_func="global"):

        super(convnext_lacunarity, self).__init__()
        self.model_dense = timm.create_model('convnext_tiny', pretrained=True, num_classes=0, global_pool='')
        model_name = Params['Model_name']
        self.feature_extraction = Params["feature_extraction"]
        self.set_parameter_requires_grad()
        
        self.avgpool = get_pooling(model_name, Params)

        self.fc = nn.Linear(num_ftrs, num_classes)
        self.agg_func = agg_func
        


    def forward(self,x):
        features = self.model_dense(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    
    def set_parameter_requires_grad(self):
        if self.feature_extraction:
            for param in self.model_dense.parameters():
                param.requires_grad = False