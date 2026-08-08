from abc import ABC, abstractmethod

import torch
from PIL import Image
from torchvision import transforms

from vision_language_clothing_retrieval.embeddings.resnet import resnet10


class ImageEncoder(ABC):
    @abstractmethod
    def encode(self, image_path: str) -> list[float]:
        """Convert an image into a vector representation."""
        raise NotImplementedError


class ResNet10ImageEncoder(ImageEncoder):
    def __init__(self) -> None:
        self.model = resnet10()
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def encode(self, image_path: str) -> list[float]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(tensor)

        return embedding.squeeze(0).tolist()
