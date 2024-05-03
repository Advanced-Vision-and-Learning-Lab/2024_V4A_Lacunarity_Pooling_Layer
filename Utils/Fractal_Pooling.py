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
    
        
class fractal_pooling(nn.Module):
    def __init__ (self, Params):

        super(fractal_pooling, self).__init__()

        self.dropout_ratio = 0.6
        self.model_name = Params["Model_name"]
        self.dense_feature_dim = Params["num_ftrs"][self.model_name]
        self.dataset = Params["Dataset"]
        self.num_classes = Params["num_classes"][self.dataset]

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1= nn.Sequential(nn.Conv2d(in_channels=self.dense_feature_dim,
                                        out_channels=self.dense_feature_dim,
                                          kernel_size=1,
                                        stride=1,
                                        padding=0),
                            nn.Dropout2d(p=self.dropout_ratio),
                              nn.BatchNorm2d(self.dense_feature_dim))
        self.sigmoid=nn.Sigmoid()
        self.relu1 = nn.Sigmoid()

        
    def forward(self, x):
        out = x
        identity = out
        identity = self.sigmoid(identity)
        out = self.conv1(out)
        out = self.relu1(out)
        out = out-identity # Residual module          
        out1 = nn.functional.adaptive_avg_pool2d(out,(1,1)).view(out.size(0), -1) 
        box_count = nn.Sequential(GDCB())
        out2 = box_count(out).view(out.size(0), -1) # Fractal pooling
        pool_layer = out1*out2
        return pool_layer