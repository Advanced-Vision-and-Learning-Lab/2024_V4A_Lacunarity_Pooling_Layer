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
from torchvision import models

## Local external libraries
from barbar import Bar
from Utils.pytorchtools import EarlyStopping
import pdb
import os
import torch.nn.functional as F
from Utils.LacunarityPoolingLayer import Pixel_Lacunarity, ScalePyramid_Lacunarity, BuildPyramid, DBC, GDCB, Base_Lacunarity
from Utils.CustomNN import Net
from Utils.Compute_sizes import get_feat_size
import matplotlib.pyplot as plt


class fusion_model(nn.Module):
    def __init__ (self, backbone, num_classes, Params):

        super(fusion_model, self).__init__()
        
        self.features=nn.Sequential(*list(backbone.features.children())[:8])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pooling_layer = backbone.avgpool
        self.classifier = backbone.classifier[2]
 
        
    def forward(self,x):
        x = self.features(x)
        x_pool = self.pooling_layer(x)
        x_avg = self.avgpool(x)
        x = x_pool * x_avg
        x = torch.flatten(x, 1)
        x = self.classifier(x)
    
        return x      

        
