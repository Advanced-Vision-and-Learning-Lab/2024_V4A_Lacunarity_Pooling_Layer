import torch
import torch.nn as nn
import numpy as np
import pdb
from kornia.geometry.transform import build_pyramid
import kornia.geometry.transform as T


global feature_maps
feature_maps =  {"resnet18_lacunarity": 512,
                "densenet161_lacunarity": 2208,
                "convnext_lacunarity": 768,
                "fusionmodel": 768}



class BuildPyramid(nn.Module):
    def __init__(self, dim=2, eps = 10E-6, model_name='resnet18_lacunarity', num_levels = None, kernel = None, stride = None, padding = None):


        # inherit nn.module
        super(BuildPyramid, self).__init__()
        # define layer properties
        self.model_name = model_name
        self.dim = dim
        self.eps = eps
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.num_levels = num_levels
        self.normalize = nn.Tanh()
        self.conv1x1 = nn.Conv2d(feature_maps[self.model_name]*self.num_levels, feature_maps[self.model_name], kernel_size=1, groups = feature_maps[self.model_name])

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


            lacunarity_values.append(L_r)
            reference_size = lacunarity_values[0].shape[-2:]
            pyr_images_resized = [T.resize(img, size=reference_size, interpolation="bilinear") for img in lacunarity_values]
        result = torch.cat(pyr_images_resized, dim=1)
        reduced_output = self.conv1x1(result)
        return reduced_output
    
