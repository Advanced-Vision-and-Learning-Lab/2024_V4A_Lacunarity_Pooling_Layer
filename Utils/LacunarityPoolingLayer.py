import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import pdb
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F


class Global_Lacunarity(nn.Module):
    def __init__(self, dim=2, eps = 10E-6, kernel = None, stride = None, padding = None):


        # inherit nn.module
        super(Global_Lacunarity, self).__init__()

        # define layer properties
        self.dim = dim
        self.eps = eps
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        #For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width
        # Time series/ signal Data
        if self.kernel is None:
            if self.dim == 1:
                self.gap_layer = nn.AdaptiveAvgPool1d(1)
            
            # Image Data
            elif self.dim == 2:
                self.gap_layer = nn.AdaptiveAvgPool2d(1)
            
            # Spatial/Temporal or Volumetric Data
            elif self.dim == 3:
                self.gap_layer = nn.AdaptiveAvgPool3d(1)
             
            else:
                raise RuntimeError('Invalid dimension for global lacunarity layer')
        else:
            if self.dim == 1:
                self.gap_layer = nn.AvgPool1d((kernel[0]), stride=stride[0], padding=(0))
            
            # Image Data
            elif self.dim == 2:
                self.gap_layer = nn.AvgPool2d((kernel[0], kernel[1]), stride=(stride[0], stride[1]), padding=(0, 0))
            
            # Spatial/Temporal or Volumetric Data
            elif self.dim == 3:
                self.gap_layer = nn.AvgPool3d((kernel[0], kernel[1], kernel[2]), stride=(stride[0], stride[1], stride[2]), padding=(0, 0, 0))
             
            else:
                raise RuntimeError('Invalid dimension for global lacunarity layer')

        
    def forward(self,x):
        #Compute squared tensor
        #pdb.set_trace()
        squared_x_tensor = x ** 2

        #Get number of samples
        n_pts = np.prod(np.asarray(self.kernel))
        if (self.kernel == None):
            n_pts = np.prod(np.asarray(x.shape[-2:]))


        #Compute numerator (n * sum of squared pixels) and denominator (squared sum of pixels)
        L_numerator = ((n_pts)**2) * (self.gap_layer(squared_x_tensor))
        L_denominator = (n_pts * self.gap_layer(x))**2

        #Lacunarity is L_numerator / L_denominator - 1
        x = (L_numerator / (L_denominator + self.eps)) - 1
        lambda_param = 0.5 #boxcox transformation
        y = (torch.pow(x.abs() + 1, lambda_param) - 1) / lambda_param
        return y
    



class CustomPoolingLayer(nn.Module):
    def __init__(self, r=3, window_size=3):
        super(CustomPoolingLayer, self).__init__()
        self.r = r
        self.window_size = window_size


    def forward(self, image):
        windows_horizontal = []
        Lr_all_horizontal = []

        for i in range(image.size(2) - self.window_size + 1):
            for j in range(image.size(3) - self.window_size + 1):
                window = image[:, :, i:i+self.window_size, j:j+self.window_size]
                windows_horizontal.append(window)

                max_pool = nn.MaxPool2d(kernel_size=window.shape[2])
                max_pool_output = F.max_pool2d(window, kernel_size=window.shape[2])
                min_pool_output = -F.max_pool2d(-window, kernel_size=window.shape[2])
                nr = torch.ceil(max_pool_output / self.r) - torch.ceil(min_pool_output / self.r) - 1
                Mr = torch.sum(nr)
                Q_mr = nr / (self.window_size - self.r + 1)
                L_r = (Mr.item()**2) * Q_mr / (Mr * Q_mr)**2
                Lr_all_horizontal.append(L_r)
                pdb.set_trace()

        return Lr_all_horizontal
    


