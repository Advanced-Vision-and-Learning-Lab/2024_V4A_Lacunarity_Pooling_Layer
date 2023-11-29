import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import pdb
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F


class Global_Lacunarity(nn.Module):
    def __init__(self, dim=2, eps = 10E-6, scales = None, kernel = None, stride = None, padding = None):


        # inherit nn.module
        super(Global_Lacunarity, self).__init__()

        # define layer properties
        self.dim = dim
        self.eps = eps
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.scales = scales
        self.normalize = nn.Tanh()
        self.conv1x1 = nn.Conv2d(len(self.scales) * 3, 3, kernel_size=1, groups = 3)

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
        lacunarity_values = []
        x = ((self.normalize(x) + 1)/2)* 255
        for scale in self.scales:
            scaled_x = x * scale
            squared_x_tensor = scaled_x ** 2

            #Get number of samples
            n_pts = np.prod(np.asarray(scaled_x.shape[-2:]))
            if (self.kernel == None):
                n_pts = np.prod(np.asarray(scaled_x.shape[-2:]))


            #Compute numerator (n * sum of squared pixels) and denominator (squared sum of pixels)
            L_numerator = ((n_pts)**2) * (self.gap_layer(squared_x_tensor))
            L_denominator = (n_pts * self.gap_layer(scaled_x))**2

            #Lacunarity is L_numerator / L_denominator - 1
            L_r = (L_numerator / (L_denominator + self.eps)) - 1
            lambda_param = 0.5 #boxcox transformation
            y = (torch.pow(L_r.abs() + 1, lambda_param) - 1) / lambda_param

            lacunarity_values.append(y)
        result = torch.cat(lacunarity_values, dim=1)
        reduced_output = self.conv1x1(result)
        return reduced_output


class Scale_Lacunarity(nn.Module):
    def __init__(self, dim=2, eps = 10E-6, scales = None, kernel = None, stride = None, padding = None):


        # inherit nn.module
        super(Scale_Lacunarity, self).__init__()

        # define layer properties
        self.dim = dim
        self.eps = eps
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.scales = scales
        self.normalize = nn.Tanh()
        self.conv1x1 = nn.Conv2d(len(self.scales) * 3, 3, kernel_size=1, groups = 3)

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
        lacunarity_values = []
        x = ((self.normalize(x) + 1)/2)* 255
        for scale in self.scales:
            scaled_x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            squared_x_tensor = scaled_x ** 2

            #Get number of samples
            n_pts = np.prod(np.asarray(scaled_x.shape[-2:]))
            if (self.kernel == None):
                n_pts = np.prod(np.asarray(scaled_x.shape[-2:]))


            #Compute numerator (n * sum of squared pixels) and denominator (squared sum of pixels)
            L_numerator = ((n_pts)**2) * (self.gap_layer(squared_x_tensor))
            L_denominator = (n_pts * self.gap_layer(scaled_x))**2

            #Lacunarity is L_numerator / L_denominator - 1
            L_r = (L_numerator / (L_denominator + self.eps)) - 1
            lambda_param = 0.5 #boxcox transformation
            y = (torch.pow(L_r.abs() + 1, lambda_param) - 1) / lambda_param

            lacunarity_values.append(y)
            print(f"Size of lacunarity values for scale {scale}: {y.size()}")
        result = torch.cat(lacunarity_values, dim=1)
        reduced_output = self.conv1x1(result)
        return reduced_output
    

class CustomPoolingLayer(nn.Module):
    def __init__(self, r=3, window_size=3, eps = 10E-6):
        super(CustomPoolingLayer, self).__init__()
        self.r = r
        self.window_size = window_size
        self.normalize = nn.Softmax2d()
        self.r_values = [0.015, 0.0625, 0.5, 0.25, 0.125, 0.2, 0.4, 0.3, 0.75, 0.6, 0.9, 0.8, 1, 2]
        self.num_output_channels = 3
        self.eps = eps
        self.conv1x1 = nn.Conv2d(len(self.r_values) * 3, self.num_output_channels, kernel_size=1, groups = 3)


    def forward(self, image):
        image = self.normalize(image) * 255
        L_r_all = []

        # Perform operations independently for each window in the current channel
        for r in self.r_values:
            max_pool = nn.MaxPool2d(kernel_size=self.window_size, stride=1)
            max_pool_output = max_pool(image)
            min_pool_output = -max_pool(-image)

            nr = torch.ceil(max_pool_output / (r + self.eps)) - torch.ceil(min_pool_output / (r + self.eps)) - 1
            Mr = torch.sum(nr)
            Q_mr = nr / (self.window_size - r + 1)
            L_r = (Mr**2) * Q_mr / (Mr * Q_mr + self.eps)**2
            L_r = L_r.squeeze(-1).squeeze(-1)
            L_r_all.append(L_r)
        channel_L_r = torch.cat(L_r_all, dim=1)
        reduced_output = self.conv1x1(channel_L_r)

        return reduced_output