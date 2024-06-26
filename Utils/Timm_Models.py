# -*- coding: utf-8 -*-
"""
Functions to train/test model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division

## PyTorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.Get_Pooling import get_pooling
import timm
import pdb

class backbone_model(nn.Module):
    def __init__(self, num_classes, Params, agg_func="global"):

        super(backbone_model, self).__init__()
        model_name = Params['Model_name']
        self.model_dense = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        num_ftrs = self.model_dense.feature_info[-1]['num_chs']
        self.feature_extraction = Params["feature_extraction"]
        self.set_parameter_requires_grad()
        
        self.avgpool = get_pooling(model_name, num_ftrs, Params)
        
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

