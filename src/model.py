import torch.nn as nn
from torchvision import models
from src.logger import get_logger
from src.exception import AppException

logger = get_logger("model")


def get_vgg16_model(num_classes, pretrained=True, freeze_features=True):
    model = models.vgg16(pretrained=pretrained)
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model
