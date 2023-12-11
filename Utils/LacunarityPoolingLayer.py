import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import pdb
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from kornia.geometry.transform import ScalePyramid, build_pyramid, resize
import kornia.geometry.transform as T


class Pixel_Lacunarity(nn.Module):
    def __init__(self, dim=2, eps = 10E-6, scales = None, kernel = None, stride = None, padding = None, bias = False):


        # inherit nn.module
        super(Pixel_Lacunarity, self).__init__()

        self.bias = bias
        # define layer properties
        self.dim = dim
        self.eps = eps
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.scales = scales
        self.normalize = nn.Tanh()
        
        
        if self.bias == False: #Non learnable parameters
            self.conv1x1 = nn.Conv2d(len(self.scales) * 3, 3, kernel_size=1, groups = 3, bias = False)
            self.conv1x1.weight.data = torch.ones(self.conv1x1.weight.shape)*1/len(self.scales)
            self.conv1x1.weight.requires_grad = False #Don't update weights
        else:
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


class ScalePyramid_Lacunarity(nn.Module):
    def __init__(self, dim=2, eps = 10E-6, num_levels = None, sigma = None, min_size = None, kernel = None, stride = None, padding = None):


        # inherit nn.module
        super(ScalePyramid_Lacunarity, self).__init__()

        # define layer properties
        self.dim = dim
        self.eps = eps
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.num_levels = num_levels
        self.sigma = sigma
        self.min_size = min_size
        self.normalize = nn.Tanh()
        self.conv1x1 = nn.Conv2d(9, 3, kernel_size=1, groups = 3)

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
        pyr_images, x, y = ScalePyramid(n_levels = self.num_levels, init_sigma = self.sigma, min_size = self.min_size)(x)

        for scaled_x in pyr_images:
            scaled_x = scaled_x[:, :, 0, :, :]
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
            reference_size = lacunarity_values[0].shape[-2:]
            pyr_images_resized = [T.resize(img, size=reference_size, interpolation="bilinear") for img in lacunarity_values]

        result = torch.cat(pyr_images_resized, dim=1)

        reduced_output = self.conv1x1(result)
        return reduced_output


class BuildPyramid(nn.Module):
    def __init__(self, dim=2, eps = 10E-6, num_levels = None, kernel = None, stride = None, padding = None):


        # inherit nn.module
        super(BuildPyramid, self).__init__()

        # define layer properties
        self.dim = dim
        self.eps = eps
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.num_levels = num_levels
        self.normalize = nn.Tanh()
        self.conv1x1 = nn.Conv2d(3*self.num_levels, 3, kernel_size=1, groups = 3)

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
        pyr_images = build_pyramid(x, max_level=self.num_levels)

        for scaled_x in pyr_images:
            squared_x_tensor = scaled_x ** 2

            #Get number of samples
            n_pts = np.prod(np.asarray(scaled_x.shape[-2:]))

            #Compute numerator (n * sum of squared pixels) and denominator (squared sum of pixels)
            L_numerator = ((n_pts)**2) * (self.gap_layer(squared_x_tensor))
            L_denominator = (n_pts * self.gap_layer(scaled_x))**2

            #Lacunarity is L_numerator / L_denominator - 1
            L_r = (L_numerator / (L_denominator + self.eps)) - 1
            lambda_param = 0.5 #boxcox transformation
            y = (torch.pow(L_r.abs() + 1, lambda_param) - 1) / lambda_param


            lacunarity_values.append(y)
            reference_size = lacunarity_values[0].shape[-2:]
            pyr_images_resized = [T.resize(img, size=reference_size, interpolation="bilinear") for img in lacunarity_values]

        result = torch.cat(pyr_images_resized, dim=1)

        reduced_output = self.conv1x1(result)
        return reduced_output

class DBC(nn.Module):
    def __init__(self, r=3, window_size=3, eps = 10E-6):
        super(DBC, self).__init__()
        self.r = r
        self.window_size = window_size
        self.normalize = nn.Tanh()
        self.r_values = [0.015, 0.0625, 0.5, 0.25, 0.125, 0.2, 0.4, 0.3, 0.75, 0.6, 0.9, 0.8, 1, 2]
        self.num_output_channels = 3
        self.eps = eps
        self.conv1x1 = nn.Conv2d(len(self.r_values) * 3, self.num_output_channels, kernel_size=1, groups = 3)


    def forward(self, image):
        image = ((self.normalize(x) + 1)/2)* 255
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