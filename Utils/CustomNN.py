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
import pdb
import os
import torch.nn.functional as F
from Utils.LacunarityPoolingLayer import Global_Lacunarity, CustomPoolingLayer


class Net(nn.Module):
    def __init__(self, num_ftrs, num_classes, pooling_layer="lacunarity", agg_func="global"):
        super(Net, self).__init__()

        self.agg_func = agg_func
        if agg_func == "global":
            if pooling_layer == "max":
                self.pooling_layer = nn.AdaptiveMaxPool2d(1)
            elif pooling_layer == "avg":
                self.pooling_layer = nn.AdaptiveAvgPool2d(1)
            elif pooling_layer == "lacunarity":
                self.pooling_layer = Global_Lacunarity()
        elif agg_func == "local":
            if pooling_layer == "max":
                self.pooling_layer = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
            elif pooling_layer == "avg":                                                                                                                                                                                                                            
                self.pooling_layer = nn.AvgPool2d((3, 3), stride=(2, 2), padding=(0, 0))
            elif pooling_layer == "lacunarity":
                self.pooling_layer = Global_Lacunarity(kernel=(3,3), stride =(2,2))
                """ self.pooling_layer = Global_Lacunarity(kernel=(3,3), stride =(2,2)) """


        self.conv1 = nn.Conv2d(3, out_channels=3, kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()

        self.fc = nn.Linear(363, num_classes)



    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x * 255
        x = self.pooling_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
