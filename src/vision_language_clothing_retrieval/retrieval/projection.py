import torch
from torch import nn


class ImageProjection(nn.Module):
    """Projektuje image embedding u zajednički embedding prostor."""

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 256,
    ) -> None:
        super().__init__()

        # Linearna projekcija preslikava ResNet embedding
        # iz 512-dimenzionalnog u zajednički prostor.
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(embeddings)


class TextProjection(nn.Module):
    """Projektuje text embedding u zajednički embedding prostor."""

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 256,
    ) -> None:
        super().__init__()

        # Linearna projekcija preslikava DistilBERT embedding
        # iz 768-dimenzionalnog u zajednički prostor.
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(embeddings)


class MultimodalProjection(nn.Module):
    """Projektuje image i text embeddinge u zajednički prostor."""

    def __init__(
        self,
        image_dim: int = 512,
        text_dim: int = 768,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()

        # Image i text imaju različite početne dimenzije,
        # pa se za svaki modalitet koristi zasebna projekcija.
        self.image_projection = ImageProjection(
            input_dim=image_dim,
            output_dim=embedding_dim,
        )

        self.text_projection = TextProjection(
            input_dim=text_dim,
            output_dim=embedding_dim,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Image embedding se preslikava u zajednički embedding prostor.
        image_embeddings = self.image_projection(
            image_embeddings
        )

        # Text embedding se preslikava u isti prostor.
        text_embeddings = self.text_projection(
            text_embeddings
        )

        # Obe reprezentacije imaju istu dimenziju i mogu se
        # direktno porediti pomoću similarity funkcija.
        return image_embeddings, text_embeddings
