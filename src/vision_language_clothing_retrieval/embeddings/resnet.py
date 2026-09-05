import torch
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)
from torchvision.models.resnet import BasicBlock, ResNet


def resnet10() -> ResNet:
    """Kreira ResNet10 sa pretrained inicijalizacijom kompatibilnih slojeva."""

    # ResNet10 arhitektura ostaje ista.
    model = ResNet(
        BasicBlock,
        [1, 1, 1, 1],
    )

    # Pretrained ResNet18 služi samo kao izvor težina.
    pretrained_model = resnet18(
        weights=ResNet18_Weights.DEFAULT,
    )

    # Učitavaju se svi kompatibilni slojevi.
    model.load_state_dict(
        pretrained_model.state_dict(),
        strict=False,
    )

    # Uklanja se klasifikacioni sloj.
    # Izlaz ostaje 512-dimenzionalni embedding.
    model.fc = torch.nn.Identity()

    return model
