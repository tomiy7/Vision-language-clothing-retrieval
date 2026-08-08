import torch
from torchvision.models.resnet import BasicBlock, ResNet


def resnet10() -> ResNet:
    model = ResNet(BasicBlock, [1, 1, 1, 1])
    model.fc = torch.nn.Identity()
    return model
