import numpy as np

from vision_language_clothing_retrieval.embeddings.image_encoder import (
    ImageEncoder,
)
from vision_language_clothing_retrieval.embeddings.text_encoder import TextEncoder


class MultimodalEncoder:
    """Kombinuje vizuelnu i tekstualnu reprezentaciju jednog uzorka."""
    def __init__(
        self,
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
    ) -> None:
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def encode(
        self,
        image_path: str,
        text: str,
    ) -> list[float]:
        # Slika i tekst se nezavisno pretvaraju u vektorske reprezentacije.
        image_embedding = self.image_encoder.encode(image_path)
        text_embedding = self.text_encoder.encode(text)

        # Dobijene reprezentacije se konkateniraju u jednu multimodalnu
        # reprezentaciju koja sadrži informacije iz obe modalnosti.
        return np.concatenate(
            [image_embedding, text_embedding]
        ).tolist()
