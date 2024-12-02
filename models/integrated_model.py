import torch
import torch.nn as nn
from models.custom_model import CustomModel

class IntegratedModel(nn.Module):
    def __init__(self, num_channels=768, num_classes=2):
        super(IntegratedModel, self).__init__()
        self.augmented_block1 = CustomModel(num_classes)
        self.final_output = nn.Conv2d(num_channels, 768, kernel_size=1)
        self.fc1 = nn.Linear(7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x_rgb, x_hsv, x_ycbcr):
        combined_input = x_rgb + x_hsv + x_ycbcr
        binary_decision = self.augmented_block1(combined_input)
        return binary_decision
