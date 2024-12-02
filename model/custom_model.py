import torch
import torch.nn as nn
from models.residual_block import ResidualBlock

class CustomModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomModel, self).__init__()
        self.residual_block = ResidualBlock(num_classes)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.residual_block(x)
        x = self.global_avg_pool(x)
        x = self.output_conv(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), -1)
        return x