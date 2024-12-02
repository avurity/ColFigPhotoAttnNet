import torch
import torch.nn as nn
from models.base_model_factory import base_model_factory
from models.integrated_model import IntegratedModel
from utils.color_space_conversion import rgb_to_hsv_tensor, rgb_to_ycbcr_tensor


class IntegratedModelWrapper(nn.Module):
    """
    A wrapper class for combining RGB, HSV, and YCbCr models into an integrated pipeline.
    Each model processes one color space, and the outputs are passed to an integrated model.
    """
    def __init__(self, base_model_factory, num_classes=2):
        super(IntegratedModelWrapper, self).__init__()
        

        self.rgb_model = base_model_factory()
        self.hsv_model = base_model_factory()
        self.ycbcr_model = base_model_factory()

        self.integrated_model = IntegratedModel(num_classes=num_classes)

    def forward(self, rgb_input):

        hsv_tensor = rgb_to_hsv_tensor(rgb_input, rgb_input.device)
        ycbcr_tensor = rgb_to_ycbcr_tensor(rgb_input, rgb_input.device)
        

        rgb_out = self.rgb_model(rgb_input)
        hsv_out = self.hsv_model(hsv_tensor)
        ycbcr_out = self.ycbcr_model(ycbcr_tensor)

        return self.integrated_model(rgb_out, hsv_out, ycbcr_out)

