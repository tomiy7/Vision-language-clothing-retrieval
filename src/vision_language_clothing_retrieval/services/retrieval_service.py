from pathlib import Path

import torch

from vision_language_clothing_retrieval.dataset.deepfashion import (
    DeepFashionDatasetLoader,
)
from vision_language_clothing_retrieval.embeddings.image_encoder import (
    ResNet10ImageEncoder,
)
from vision_language_clothing_retrieval.embeddings.text_encoder import (
    DistilBERTTextEncoder,
)
from vision_language_clothing_retrieval.retrieval.multimodal_retriever import (
    MultimodalRetriever,
)


class RetrievalService:

    def __init__(
        self,
        model_path: str = "embeddings/multimodal_model.pt",
        embeddings_path: str = "embeddings/test.pt",
        image_dir: str = "notebooks/data/images",
        captions_path: str = "notebooks/data/captions.json",
        device: str = "cpu",
    ) -> None:
        self.device = device

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Fajl modela nije pronađen: {model_path}"
            )

        if not Path(embeddings_path).exists():
            raise FileNotFoundError(
                f"Fajl sa embeddingima nije pronađen: {embeddings_path}"
            )

        if not Path(captions_path).exists():
            raise FileNotFoundError(
                f"Fajl sa opisima nije pronađen: {captions_path}"
            )

        if not Path(image_dir).exists():
            raise FileNotFoundError(
                f"Direktorijum sa slikama nije pronađen: {image_dir}"
            )

        self.retriever = MultimodalRetriever(
            model_path=model_path,
            device=device,
        )

        self.text_encoder = DistilBERTTextEncoder()
        self.image_encoder = ResNet10ImageEncoder()

        embeddings = torch.load(
            embeddings_path,
            map_location=device,
            weights_only=True,
        )

        self.image_embeddings = embeddings["image_embeddings"]
        self.text_embeddings = embeddings["text_embeddings"]
        self.sample_ids = embeddings["sample_ids"]

        loader = DeepFashionDatasetLoader(
            image_dir=image_dir,
            captions_path=captions_path,
        )

        self.samples = {
            sample.sample_id: sample
            for sample in loader.load()
        }

    def retrieve_images(
        self,
        text: str,
        top_k: int = 5,
    ) -> list[dict]:

        self._validate_top_k(top_k)

        text_embedding = torch.tensor(
            self.text_encoder.encode(text),
            dtype=torch.float,
        )

        indices, scores = self.retriever.retrieve_image(
            text_embedding=text_embedding,
            image_embeddings=self.image_embeddings,
            top_k=min(top_k, len(self.image_embeddings)),
        )

        results = []

        for index, score in zip(indices, scores):
            sample_id = self.sample_ids[index.item()]
            sample = self.samples[sample_id]

            results.append(
                {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "text": sample.text,
                    "score": score.item(),
                }
            )

        return results

    def retrieve_texts(
        self,
        image_path: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Za datu sliku pronalazi najsličnije tekstualne opise."""

        self._validate_top_k(top_k)

        if not Path(image_path).exists():
            raise FileNotFoundError(
                f"Slika nije pronađena: {image_path}"
            )

        image_embedding = torch.tensor(
            self.image_encoder.encode(image_path),
            dtype=torch.float,
        )

        indices, scores = self.retriever.retrieve_text(
            image_embedding=image_embedding,
            text_embeddings=self.text_embeddings,
            top_k=min(top_k, len(self.text_embeddings)),
        )

        results = []

        for index, score in zip(indices, scores):
            sample_id = self.sample_ids[index.item()]
            sample = self.samples[sample_id]

            results.append(
                {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "text": sample.text,
                    "score": score.item(),
                }
            )

        return results

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k mora biti veći od 0.")