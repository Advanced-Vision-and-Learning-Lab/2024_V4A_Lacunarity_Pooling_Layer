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
from Utils.CustomNN import Net
from Utils.Compute_sizes import get_feat_size
import matplotlib.pyplot as plt
import timm

import torch
import torch.nn as nn
import torch.optim as optimxxc
import numpy as np
import os
from torchvision import models, transforms
import math

class GDCB(nn.Module):
    def __init__(self,mfs_dim=25,nlv_bcd=6):
        super(GDCB,self).__init__()
        self.mfs_dim=mfs_dim
        self.nlv_bcd=nlv_bcd
        self.pool=nn.ModuleList()
        
        for i in range(self.nlv_bcd-1):
            self.pool.add_module(str(i),nn.MaxPool2d(kernel_size=i+2,stride=(i+2)//2))
        self.ReLU = nn.ReLU()
        
    def forward(self,input):
        tmp=[]
        for i in range(self.nlv_bcd-1):
            output_item=self.pool[i](input)
            tmp.append(torch.sum(torch.sum(output_item,dim=2,keepdim=True),dim=3,keepdim=True))
        output=torch.cat(tuple(tmp),2)
        output=torch.log2(self.ReLU(output)+1)
        X=[-math.log(i+2,2) for i in range(self.nlv_bcd-1)]
        X = torch.tensor(X).to(output.device)
        X=X.view([1,1,X.shape[0],1])
        meanX = torch.mean(X,2,True)
        meanY = torch.mean(output,2,True)
        Fracdim = torch.div(torch.sum((output-meanY)*(X-meanX),2,True),torch.sum((X-meanX)**2,2,True))
        return Fracdim
    
class fusion_model(nn.Module):
    def __init__ (self, model_name, backbone, num_classes, Params):

        super(fusion_model, self).__init__()
        self.model_name = model_name
        poolingLayer = Params['pooling_layer']
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #lacunarity layer is added in backbone
        
        if poolingLayer == "Baseline":
            if self.model_name == 'efficientnet_lacunarity':
                self.features=nn.Sequential(*list(backbone.children())[:-2])
                self.classifier = backbone.classifier
                self.pooling_layer = backbone.avgpool
        
        else: 
            if self.model_name == 'convnext_lacunarity':
                self.features=nn.Sequential(*list(backbone.features.children())[:8])
                self.classifier = backbone.classifier[2]
                self.pooling_layer = backbone.avgpool

            elif self.model_name == 'resnet18_lacunarity':
                self.features=nn.Sequential(*list(backbone.children())[:-2])
                self.classifier = backbone.fc
                self.pooling_layer = backbone.avgpool

            elif self.model_name == 'densenet161_lacunarity':
                self.features=backbone.features 
                self.classifier = backbone.classifier
                self.pooling_layer = backbone.avgpool

            
            elif self.model_name == 'efficientnet_lacunarity':
                self.features=nn.Sequential(*list(backbone.children())[:-3])
                self.classifier = backbone.classifier
                self.pooling_layer = backbone.avgpool
 
        
    def forward(self, x):
        x = self.features(x)
        x_pool = self.pooling_layer(x)
        x_avg = self.avgpool(x)
        x = x_pool * x_avg
        x = torch.flatten(x, 1)
        x = self.classifier(x)
    
        return x      

        
class fractal_model(nn.Module):
    def __init__ (self, model_name, backbone, num_classes, Params):

        super(fractal_model, self).__init__()
        self.backbone = backbone
        self.model_name = model_name
        if self.model_name == "densenet161_lacunarity":
            dense_feature_dim = 2208
        elif self.model_name == "convnext_lacunarity":
            dense_feature_dim = 768        
        elif self.model_name == "resnet18_lacunarity":
            dense_feature_dim = 512
        elif self.model_name == "efficientnet_lacunarity":
            dense_feature_dim = 1280
        dropout_ratio = 0.6
        self.poolingLayer = Params['pooling_layer']

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1= nn.Sequential(nn.Conv2d(in_channels=dense_feature_dim,
                                        out_channels=dense_feature_dim,
                                          kernel_size=1,
                                        stride=1,
                                        padding=0),
                            nn.Dropout2d(p=dropout_ratio),
                              nn.BatchNorm2d(dense_feature_dim))
        self.sigmoid=nn.Sigmoid()
        self.relu1 = nn.Sigmoid()

        if self.poolingLayer == "Baseline":
            if self.model_name == 'efficientnet_lacunarity':
                self.features=nn.Sequential(*list(self.backbone.children())[:-2])
                self.classifier = self.backbone.classifier
                self.pooling_layer = self.backbone.avgpool
            
        
        else: 
            if self.model_name == 'efficientnet_lacunarity':
                self.features=nn.Sequential(*list(self.backbone.children())[:-3])
                self.classifier = self.backbone.classifier
                self.pooling_layer = self.backbone.avgpool


        if self.model_name == 'convnext_lacunarity':
            self.features=nn.Sequential(*list(self.backbone.features.children())[:8])
            self.classifier = self.backbone.classifier
            self.pooling_layer = self.backbone.avgpool

        elif self.model_name == 'resnet18_lacunarity':
            self.features=nn.Sequential(*list(self.backbone.children())[:-2])
            self.classifier = self.backbone.fc
            self.pooling_layer = self.backbone.avgpool

        elif self.model_name == 'densenet161_lacunarity':
            self.features=self.backbone.features
            self.classifier = self.backbone.classifier
            self.pooling_layer = self.backbone.avgpool

        
    def forward(self,x):
        out = self.features(x)
        identity=out
        identity = self.sigmoid(identity)                
        out = self.conv1(out)
        out = self.relu1(out)
        out = out-identity # Residual module          
        out1 = nn.functional.adaptive_avg_pool2d(out,(1,1)).view(out.size(0), -1) 
        box_count = nn.Sequential(GDCB())
        out2 = box_count(out).view(out.size(0), -1) # Fractal pooling
        out3 = out1*out2
        x=self.classifier(out3)
        return x