# -*- coding: utf-8 -*-
"""
Functions to train/test model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import math


def get_feat_size(Params, dataloaders):
    kernel = Params["kernel"]
    stride = Params["stride"]
    model_name = Params["Model_name"]
    pooling_layer = Params["pooling_layer"]
    agg_func = Params["agg_func"]

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
    
    elif model_name == "resnet18":
        if agg_func == "local":
            feature_height = 7
            out_channels = 512
            pooling_output = math.floor((feature_height - kernel) / stride) + 1
            num_ftrs = out_channels * pooling_output * pooling_output
        elif agg_func == "global":
            if pooling_layer == "DBC":
                num_ftrs = 512
            else:
                num_ftrs = 512
    
    elif model_name == "convnext_tiny":
        
        if agg_func == "local":
            feature_height = 7
            out_channels = 768
            pooling_output = math.floor((feature_height - kernel) / stride) + 1
            num_ftrs = out_channels * pooling_output * pooling_output
        elif agg_func == "global":
            if pooling_layer == "DBC":
                num_ftrs = 768
            else:
                num_ftrs = 768

    elif model_name == "densenet161":
        if agg_func == "local":
            feature_height = 7
            out_channels = 2208
            pooling_output = math.floor((feature_height - kernel) / stride) + 1
            num_ftrs = out_channels * pooling_output * pooling_output
        elif agg_func == "global":
            if pooling_layer == "DBC":
                num_ftrs = 2208
            else:
                num_ftrs = 2208


    return num_ftrs