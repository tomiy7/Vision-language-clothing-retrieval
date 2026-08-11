from abc import ABC, abstractmethod

import torch
from PIL import Image
from torchvision import transforms

from vision_language_clothing_retrieval.embeddings.resnet import resnet10


class ImageEncoder(ABC):
    """Interfejs za pretvaranje slike u vektorsku reprezentaciju."""
    @abstractmethod
    def encode(self, image_path: str) -> list[float]:
        raise NotImplementedError


class ResNet10ImageEncoder(ImageEncoder):
    """Image encoder zasnovan na ResNet10 modelu."""
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
        # Slika se učitava kao RGB kako bi sve slike imale isti broj kanala.
        image = Image.open(image_path).convert("RGB")

        # Primena definisanih transformacija slike.
        tensor = self.transform(image)

        # Model očekuje batch ulaz, pa se dodaje dimenzija batch-a.
        tensor = tensor.unsqueeze(0)

        # Prilikom generisanja embeddinga nisu potrebni gradijenti,
        # jer se model koristi samo za inference.
        with torch.no_grad():
            embedding = self.model(tensor)

        # Uklanja se batch dimenzija i embedding se pretvara u Python listu.
        return embedding.squeeze(0).tolist()
