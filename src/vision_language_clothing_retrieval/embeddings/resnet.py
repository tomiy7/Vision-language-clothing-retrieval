import torch
from torchvision.models.resnet import BasicBlock, ResNet


def resnet10() -> ResNet:
    """Kreira ResNet10 model koji se koristi za ekstrakciju image embeddinga."""

    # Uklanja se završni klasifikacioni sloj kako bi izlaz modela
    # predstavljao 512-dimenzionalnu vizuelnu reprezentaciju slike.
    model = ResNet(BasicBlock, [1, 1, 1, 1])
    model.fc = torch.nn.Identity()
    return model
