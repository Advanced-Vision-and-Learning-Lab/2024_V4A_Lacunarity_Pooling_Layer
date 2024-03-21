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
from Utils.LacunarityPoolingLayer import Pixel_Lacunarity, ScalePyramid_Lacunarity, BuildPyramid, DBC, GDCB, Base_Lacunarity


class Net(nn.Module):
    def __init__(self, num_ftrs, num_classes, Params, pooling_layer="lacunarity", agg_func="global"):

        super(Net, self).__init__()

        model_name = Params['Model_name']
        kernel = Params["kernel"]
        stride = Params["stride"]
        padding = Params["conv_padding"]
        scales = Params["scales"]
        num_levels = Params["num_levels"]
        sigma = Params["sigma"]
        min_size = Params["min_size"]
        bias = Params["bias"]

        self.agg_func = agg_func
        if agg_func == "global":
            if pooling_layer == "max":
                self.pooling_layer = nn.AdaptiveMaxPool2d(1)
            elif pooling_layer == "avg":
                self.pooling_layer = nn.AdaptiveAvgPool2d(1)
            elif pooling_layer == "Pixel_Lacunarity":
                self.pooling_layer = Pixel_Lacunarity(scales=scales, bias = bias)
            elif pooling_layer == "Base_Lacunarity":
                self.pooling_layer = Base_Lacunarity(model_name=model_name, scales=scales,bias=bias)
        elif agg_func == "local":
            if pooling_layer == "max":
                self.pooling_layer = nn.MaxPool2d(kernel_size=(kernel, kernel), stride =(stride, stride), padding=(padding, padding))
            elif pooling_layer == "avg":                                                                                                                                                                                                                           
                self.pooling_layer = nn.AvgPool2d(kernel_size=(kernel, kernel), stride =(stride, stride), padding=(padding, padding))
            elif pooling_layer == "Base_Lacunarity":
                self.pooling_layer = Base_Lacunarity(model_name=model_name, scales=scales, kernel=(kernel, kernel), stride =(stride, stride), bias=bias)
            elif pooling_layer == "Pixel_Lacunarity":
                self.pooling_layer = Pixel_Lacunarity(model_name=model_name, scales=scales, kernel=(kernel, kernel), stride =(stride, stride), bias=bias)
            elif pooling_layer == "ScalePyramid_Lacunarity":
                self.pooling_layer = ScalePyramid_Lacunarity(model_name, num_levels=num_levels, sigma = sigma, min_size = min_size, kernel=(kernel, kernel), stride =(stride, stride))
            elif pooling_layer == "BuildPyramid":
                self.pooling_layer = BuildPyramid(model_name, num_levels=num_levels, kernel=(kernel, kernel), stride =(stride, stride))
            elif pooling_layer == "DBC":
                self.pooling_layer = DBC(model_name, r_values = scales, window_size = kernel)
            elif pooling_layer == "GDCB":
                self.pooling_layer = GDCB(3,5)

                """Scale_Lacunarity(kernel=(3,3), stride =(1,1))"""
                """ self.pooling_layer = Global_Lacunarity(scales=[i/10.0 for i in range(0, 20)], kernel=(4,4), stride =(1,1)) """
                """[i/10 for i in range(10, 21)]"""
                """[i/10 for i in range(10, 31)]"""


        self.conv1 = nn.Conv2d(3, out_channels=3, kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.fc = nn.Linear(num_ftrs, num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x_pool = self.pooling_layer(x)
        _, _, h, w = x_pool.shape
        x_avg = nn.functional.adaptive_avg_pool2d(x, (h,w))
        x = x_pool * x_avg
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    


class densenet161_lacunarity(nn.Module):
    def __init__(self, num_ftrs, num_classes, Params, pooling_layer="lacunarity", agg_func="global"):

        super(densenet161_lacunarity, self).__init__()
        model_dense=models.densenet161(pretrained=True)
        self.features=model_dense.features
        self.classifier = model_dense.classifier

        model_name = Params['Model_name']
        kernel = Params["kernel"]
        stride = Params["stride"]
        padding = Params["conv_padding"]
        scales = Params["scales"]
        num_levels = Params["num_levels"]
        sigma = Params["sigma"]
        min_size = Params["min_size"]
        bias = Params["bias"]
        
        if pooling_layer == "Baseline":
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.agg_func = agg_func
        if self.agg_func == "local":
            if pooling_layer == "max":
                self.avgpool = nn.MaxPool2d(kernel_size=(kernel, kernel), stride =(stride, stride), padding=(padding, padding))
            elif pooling_layer == "avg":                                                                                                                                                                                                                            
                self.avgpool = nn.AvgPool2d(kernel_size=(kernel, kernel), stride =(stride, stride), padding=(padding, padding))
            elif pooling_layer == "Base_Lacunarity":
                self.avgpool = Base_Lacunarity(model_name=model_name, scales=scales, kernel=(kernel, kernel), stride =(stride, stride), bias=bias)
            elif pooling_layer == "Pixel_Lacunarity":
                self.avgpool = Pixel_Lacunarity(model_name=model_name, scales=scales, kernel=(kernel, kernel), stride =(stride, stride), bias=bias)
            elif pooling_layer == "ScalePyramid_Lacunarity":
                self.avgpool = ScalePyramid_Lacunarity(model_name=model_name, num_levels=num_levels, sigma = sigma, min_size = min_size, kernel=(kernel, kernel), stride =(stride, stride))
            elif pooling_layer == "BuildPyramid":
                self.avgpool = BuildPyramid(model_name=model_name, num_levels=num_levels, kernel=(kernel, kernel), stride =(stride, stride))
            elif pooling_layer == "DBC":
                self.avgpool = DBC(model_name=model_name, r_values = scales, window_size = kernel)
            elif pooling_layer == "GDCB":
                self.avgpool = GDCB(3,5)
        
        elif self.agg_func == "global":
            if pooling_layer == "max":
                self.avgpool = nn.AdaptiveMaxPool2d((1,1))
            elif pooling_layer == "avg":                                                                                                                                                                                                                            
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            elif pooling_layer == "L2":                                                                                                                                                                                                                           
                self.avgpool = nn.LPPool2d(norm_type=2, kernel_size=(7, 7))
            elif pooling_layer == "Base_Lacunarity":
                self.avgpool = Base_Lacunarity(model_name=model_name, scales=scales,bias=bias)
            elif pooling_layer == "Pixel_Lacunarity":
                self.avgpool = Pixel_Lacunarity(model_name=model_name, scales=scales, bias=bias)
            elif pooling_layer == "ScalePyramid_Lacunarity":
                self.avgpool = ScalePyramid_Lacunarity(model_name=model_name, num_levels=num_levels, sigma = sigma, min_size = min_size)
            elif pooling_layer == "BuildPyramid":
                self.avgpool = BuildPyramid(model_name=model_name, num_levels=num_levels)
            elif pooling_layer == "DBC":
                self.avgpool = DBC(model_name=model_name, r_values = scales, window_size = 8)
            elif pooling_layer == "GDCB":
                self.avgpool = GDCB(3,5)


    def forward(self,x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
