import torch
import torch.nn as nn

class ChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelReducer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Apply convolution
        conv_output = self.conv(x)
        # Reshape and transpose the output
        reshaped_output = conv_output.view(x.size(0), -1, conv_output.size(1))
        return reshaped_output