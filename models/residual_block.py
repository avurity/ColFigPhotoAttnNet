import torch
import torch.nn as nn
from torchvision import models
from models.reshape_and_conv import ReshapeAndConv

class ResidualBlock(nn.Module):
    def __init__(self, num_classes):
        super(ResidualBlock, self).__init__()
        # Load the pretrained ResNet-34 model
        resnet = models.resnet34(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = True
        n_inputs_resnet = resnet.fc.in_features
        resnet.fc = nn.Linear(n_inputs_resnet, num_classes)
        model_address = 'pth/res34.pth'
        resnet.load_state_dict(torch.load(model_address, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        # Extract the layer 4 residual block
        self.residual_block = resnet.layer4
        self.reshape_and_conv = ReshapeAndConv()

    def forward(self, x):
        x = self.reshape_and_conv(x)
        return self.residual_block(x)
