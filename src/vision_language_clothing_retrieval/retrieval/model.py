import torch
from torch import nn

from vision_language_clothing_retrieval.retrieval.projection import (
    ImageProjection,
    TextProjection,
)


class MultimodalModel(nn.Module):
    """Model koji image i text embeddinge projektuje u zajednički prostor."""

    def __init__(
        self,
        image_input_dim: int = 512,
        text_input_dim: int = 768,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()

        # Image embedding se projektuje iz originalnog prostora od 512
        # dimenzija u zajednički embedding prostor.
        self.image_projection = ImageProjection(
            input_dim=image_input_dim,
            output_dim=embedding_dim,
        )

        # Text embedding se projektuje iz originalnog prostora od 768
        # dimenzija u isti zajednički embedding prostor.
        self.text_projection = TextProjection(
            input_dim=text_input_dim,
            output_dim=embedding_dim,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Image embeddingi se preslikavaju u zajednički prostor.
        image_embeddings = self.image_projection(
            image_embeddings
        )

        # Text embeddingi se preslikavaju u isti zajednički prostor.
        text_embeddings = self.text_projection(
            text_embeddings
        )

        # Vraćaju se obe reprezentacije kako bi mogle da se
        # porede pomoću similarity funkcije ili koriste za loss.
        return image_embeddings, text_embeddings
