from __future__ import absolute_import

from .resnet import resnet18, resnet34
from .wideresnet import wrn34

from torchvision import models
import torch.nn as nn


model_dict = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'wrn34': wrn34,
}

def get_model(args, num_classes, **kwargs):
    model_ft = model_dict[args.arch](pretrained=args.pretrained)
    
    if args.feature_extraction:
        # freeze all except the final layer
        for param in list(model_ft.parameters())[:-2]:
            param.requires_grad = False
    
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

#     return model_dict[args.arch](num_classes, **kwargs)
