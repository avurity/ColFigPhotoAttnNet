import torch
import torch.nn as nn
import torch.nn.functional as F

class ReshapeAndConv(nn.Module):
    def __init__(self):
        super(ReshapeAndConv, self).__init__()
        # Define a 1x1 convolution to reduce channel dimension from 768 to 256
        self.conv1x1 = nn.Conv2d(768, 256, kernel_size=1)

    def forward(self, x):
        # Reshape from [1, 1, 49, 768] to [1, 768, 7, 7]
        x = x.view(x.size(0), 768, 7, 7)
        # Apply 1x1 convolution to reduce channels and maintain spatial dimensions
        x = self.conv1x1(x)
        # Upsample from [1, 256, 7, 7] to [1, 256, 14, 14]
        x = F.interpolate(x, size=(14, 14), mode='nearest')
        return x
