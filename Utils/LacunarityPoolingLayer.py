import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import pdb

class Global_Lacunarity(nn.Module):
    def __init__(self,dim=2):

        # inherit nn.module
        super(Global_Lacunarity, self).__init__()

        # define layer properties
        #self.in_channels = in_channels
        self.dim = dim

        #For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width
        # Time series/ signal Data
        if self.dim == 1:
            self.gap_layer = nn.AdaptiveAvgPool2d(1)
        
        # Image Data
        elif self.dim == 2:
            self.gap_layer = nn.AdaptiveAvgPool2d(1)
        
        # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            self.gap_layer = nn.AdaptiveAvgPool3d(1)
            
        else:
            raise RuntimeError('Invalid dimension for global lacunarity layer')
        
    def forward(self,x):        
        
        #Compute squared tensor
        pdb.set_trace()
        squared_x_tensor = x ** 2

        #Get number of samples 
        n_pts = np.prod(np.asarray(x.shape[-2:]))

        #Compute numerator (n * sum of squared pixels) and denominator (squared sum of pixels)
        L_numerator = ((n_pts)**2) * (self.gap_layer(squared_x_tensor))
        L_denominator = (n_pts * self.gap_layer(x))**2

        #Lacunarity is L_numerator / L_denominator - 1
        x = (L_numerator / L_denominator) - 1

        return x
    
if __name__ == "__main__":

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    #load image, if you use grayscale iamge, cosine similarity will be 1 (input dim = 1)
    X = np.array( [[25, 41, 24],[29, 33, 80],[38, 56, 65]], dtype=np.int16)
    
    #Convert to Pytorch tensor (Batch x num channels x height x width)
    X = transforms.ToTensor()(X).unsqueeze(0).float()
    
    #Compute similarity feature (currenlty have 'norm' or 'cosine')
    channels = X.shape[1]
    Lancunarity_Layer = Global_Lacunarity()
    similarity_features = Lancunarity_Layer(X)
    

        
    
    