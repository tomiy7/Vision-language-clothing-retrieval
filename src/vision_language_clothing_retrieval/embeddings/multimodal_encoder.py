import numpy as np

from vision_language_clothing_retrieval.embeddings.image_encoder import (
    ImageEncoder,
)
from vision_language_clothing_retrieval.embeddings.text_encoder import TextEncoder


class MultimodalEncoder:
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
        image_embedding = self.image_encoder.encode(image_path)
        text_embedding = self.text_encoder.encode(text)

        return np.concatenate(
            [image_embedding, text_embedding]
        ).tolist()
