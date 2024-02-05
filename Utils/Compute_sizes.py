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
from Utils.LacunarityPoolingLayer import Pixel_Lacunarity
import math


def get_feat_size(model_name, Params, pooling_layer, agg_func, dataloaders):
    kernel = Params["kernel"]
    stride = Params["stride"]

    if model_name == "Net":
        if agg_func == "global":
            num_ftrs = 3
        else:
    #output size = [(input size - kernel size) / stride] + 1
            for idx, (inputs, labels, index) in enumerate(dataloaders['train']):
                batch, channels, h, w = inputs.shape
                out_channels = 3
                conv_parameters = math.floor(((h - 3) / 2) + 1) #conv2d
                #relu parameters remains the same
                #self.pooling layer
                pooling_output = math.floor((conv_parameters - kernel) / stride) + 1
                num_ftrs = out_channels * pooling_output * pooling_output
    
    elif model_name == "resnet18_lacunarity":
        if agg_func == "local":
            feature_height = 7
            out_channels = 512
            pooling_output = math.floor((feature_height - kernel) / stride) + 1
            num_ftrs = out_channels * pooling_output * pooling_output
        elif agg_func == "global":
            num_ftrs = 512

    return num_ftrs