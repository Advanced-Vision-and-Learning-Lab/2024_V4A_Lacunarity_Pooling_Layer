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
import torch.nn as nn

## Local external libraries
from torchvision import models
import pdb
import os
from Utils.Base_Lacunarity import Base_Lacunarity
from Utils.Multi_Scale_Lacunarity import BuildPyramid
from Utils.DBC import DBC


def get_pooling(model_name, Params):
    kernel = Params["kernel"]
    stride = Params["stride"]
    padding = Params["conv_padding"]
    scales = Params["scales"]
    num_levels = Params["num_levels"]
    bias = Params["bias"]
    poolingLayer = Params["pooling_layer"]
    aggFunc = Params["agg_func"]

    if aggFunc == "local":
        if poolingLayer == "max":
            avgpool = nn.MaxPool2d(kernel_size=(kernel, kernel), stride =(stride, stride), padding=(padding, padding))
        elif poolingLayer == "avg":                                                                                                                                                                                                                            
            avgpool = nn.AvgPool2d(kernel_size=(kernel, kernel), stride =(stride, stride), padding=(padding, padding))
        elif poolingLayer == "L2":                                                                                                                                                                                                                           
            avgpool = nn.LPPool2d(norm_type=2, kernel_size=(kernel, kernel))
        elif poolingLayer == "Base_Lacunarity":
            avgpool = Base_Lacunarity(model_name=model_name, scales=scales, kernel=(kernel, kernel), stride =(stride, stride), bias=bias)
        elif poolingLayer == "BuildPyramid":
            avgpool = BuildPyramid(model_name=model_name, num_levels=num_levels, kernel=(kernel, kernel), stride =(stride, stride))
        elif poolingLayer == "DBC":
            avgpool = DBC(model_name=model_name, r_values = scales, window_size = kernel)
    
    elif aggFunc == "global":
        if poolingLayer == "max":
            avgpool = nn.AdaptiveMaxPool2d((1,1))
        elif poolingLayer == "avg":                                                                                                                                                                                                                            
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif poolingLayer == "L2":                                                                                                                                                                                                                           
            avgpool = nn.LPPool2d(norm_type=2, kernel_size=(7, 7))
        elif poolingLayer == "Base_Lacunarity":
            avgpool = Base_Lacunarity(model_name=model_name, scales=scales, bias=bias)
        elif poolingLayer == "BuildPyramid":
            avgpool = BuildPyramid(model_name=model_name, num_levels=num_levels)
        elif poolingLayer == "DBC":
            avgpool = DBC(model_name=model_name, r_values = scales, window_size = 7)

    return avgpool