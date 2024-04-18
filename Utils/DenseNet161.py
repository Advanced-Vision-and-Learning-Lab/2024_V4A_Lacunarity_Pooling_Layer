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


class densenet161_lacunarity(nn.Module):
    def __init__(self, num_ftrs, num_classes, Params, pooling_layer="lacunarity", agg_func="global"):

        super(densenet161_lacunarity, self).__init__()
        model_dense=models.densenet161(pretrained=True)
        self.features=model_dense.features
        self.classifier = model_dense.classifier
        self.feature_extraction = Params["feature_extraction"]

        model_name = Params['Model_name']
        self.set_parameter_requires_grad()
        
        if pooling_layer == "Baseline":
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.agg_func = agg_func
        self.avgpool = get_pooling(model_name, Params)


    def forward(self,x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def set_parameter_requires_grad(self):
        if self.feature_extraction:
            for param in self.features.parameters():
                param.requires_grad = False
