import torch
import torch.nn as nn
import pdb


class DBC_Lacunarity(nn.Module):
    def __init__(self, model_name='Net', window_size=3, eps = 10E-6):
        super(DBC_Lacunarity, self).__init__()
        self.window_size = window_size
        self.normalize = nn.Tanh()
        self.num_output_channels = 3
        self.eps = eps
        self.r = 1
        self.model_name = model_name
        self.max_pool = nn.MaxPool2d(kernel_size=self.window_size, stride=1) #feature map is 7 for classifiers
        
    def forward(self, image):
        L_r_all = []
        image = ((self.normalize(image) + 1)/2)* 255

        # Perform operations independently for each window in the current channel
        max_pool_output = self.max_pool(image)
        min_pool_output = -self.max_pool(-image)

        nr = torch.ceil(max_pool_output / (self.r + self.eps)) - torch.ceil(min_pool_output / (self.r + self.eps)) - 1
        Mr = torch.sum(nr)
        Q_mr = nr / (self.window_size - self.r + 1)
        L_r = (Mr**2) * Q_mr / (Mr * Q_mr + self.eps)**2
        L_r_all.append(L_r)
        channel_L_r = torch.cat(L_r_all, dim=1)
        return channel_L_r
