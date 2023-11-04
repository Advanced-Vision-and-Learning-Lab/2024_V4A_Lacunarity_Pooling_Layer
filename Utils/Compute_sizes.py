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
from Utils.LacunarityPoolingLayer import Global_Lacunarity
import math


def get_feat_size(pooling_layer, agg_func, dataloaders):
    if agg_func == "global":
        num_ftrs = 3

    #output size = [(input size - kernel size) / stride] + 1
    else:
        for idx, (inputs, labels, index) in enumerate(dataloaders['train']):
            batch, channels, h, w = inputs.shape
            conv_parameters = math.floor(((h - 3) / 2) + 1) #conv2d
            #relu parameters remains the same
            #self.pooling layer
            pooling_output = math.floor((conv_parameters - 3) / 2) + 1
            num_ftrs = channels * pooling_output * pooling_output

    return num_ftrs