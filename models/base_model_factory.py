import torch
from torchvision import models
from models.channel_reducer import ChannelReducer
import torch.nn as nn

def base_model_factory(in_channels=960, num_classes=2):
    model_address = "pth/swin.pth"
    HUB_URL = "SharanSMenon/swin-transformer-hub:main"
    MODEL_NAME = "swin_tiny_patch4_window7_224"
    swin_transformer = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True)

    n_inputs = swin_transformer.head.in_features
    swin_transformer.head = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    swin_transformer.load_state_dict(torch.load(model_address, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    for param in swin_transformer.parameters():
        param.requires_grad = True

    mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
    mobilenet_v3_large.classifier = nn.Identity()
    for param in mobilenet_v3_large.features.parameters():
        param.requires_grad = True

    return nn.Sequential(
        mobilenet_v3_large.features,
        ChannelReducer(960, 768),
        swin_transformer.layers[-1].blocks[-1].attn,
    )
