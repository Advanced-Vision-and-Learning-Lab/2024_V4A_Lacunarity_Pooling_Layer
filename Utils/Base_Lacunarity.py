import torch
import torch.nn as nn
import numpy as np
import pdb


class Base_Lacunarity(nn.Module):
    def __init__(self, dim=2, eps = 10E-6, model_name=None, kernel = None, stride = None, padding = None):

        # inherit nn.module
        super(Base_Lacunarity, self).__init__()


        # define layer properties
        self.dim = dim
        self.eps = eps
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        
        self.model_name = model_name
        self.normalize = nn.Tanh()
        

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
        squared_x_tensor = x ** 2

        #Get number of samples
        n_pts = np.prod(np.asarray(x.shape[-2:]))
        if (self.kernel == None):
            n_pts = np.prod(np.asarray(x.shape[-2:]))

        #Compute numerator (n * sum of squared pixels) and denominator (squared sum of pixels)
        L_numerator = ((n_pts)**2) * (self.gap_layer(squared_x_tensor))
        L_denominator = (n_pts * self.gap_layer(x))**2

        #Lacunarity is L_numerator / L_denominator - 1
        L_r = (L_numerator / (L_denominator + self.eps)) - 1
        lacunarity_values.append(L_r)
        result = torch.cat(lacunarity_values, dim=1)
        return result
