import torch

from vision_language_clothing_retrieval.retrieval.model import (
    MultimodalModel,
)
from vision_language_clothing_retrieval.retrieval.retriever import (
    Retriever,
)


class MultimodalRetriever:
    """Retrieval nad naučenim multimodalnim projekcijama."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
    ) -> None:
        # Uređaj na kome će se izvršavati inference.
        self.device = device

        # Kreira se model iste arhitekture koji je korišćen tokom treninga.
        self.model = MultimodalModel()

        # Učitavaju se naučeni parametri projekcionih slojeva.
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
                weights_only=True,
            )
        )

        # Model se prebacuje na izabrani uređaj i postavlja u evaluation
        # režim jer se koristi samo za inference.
        self.model.to(device)
        self.model.eval()

        # Retriever obavlja pronalaženje najsličnijih embeddinga.
        self.retriever = Retriever()

    def project_embeddings(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Embeddingi se prebacuju na isti uređaj kao model.
        image_embeddings = image_embeddings.to(self.device)
        text_embeddings = text_embeddings.to(self.device)

        # Tokom inference-a nisu potrebni gradijenti.
        with torch.no_grad():
            projected_images, projected_texts = self.model(
                image_embeddings,
                text_embeddings,
            )

        # Oba modaliteta sada se nalaze u zajedničkom 256-dimenzionalnom
        # embedding prostoru.
        return projected_images, projected_texts

    def retrieve_text(
            self,
            image_embedding: torch.Tensor,
            text_embeddings: torch.Tensor,
            top_k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Za dati image embedding pronalazi najsličnije text embeddinge."""

        # Image i text embeddingi se projektuju u zajednički embedding prostor.
        projected_image, projected_texts = self.project_embeddings(
            image_embedding,
            text_embeddings,
        )

        # Postojeći Retriever pronalazi top-k tekstova na osnovu
        # cosine similarity vrednosti.
        indices, scores = self.retriever.retrieve_text(
            projected_image.squeeze(0),
            projected_texts,
            top_k=top_k,
        )

        return indices, scores

    def retrieve_image(
            self,
            text_embedding: torch.Tensor,
            image_embeddings: torch.Tensor,
            top_k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Za dati text embedding pronalazi najsličnije image embeddinge."""

        # Text i image embeddingi se projektuju u zajednički embedding prostor.
        projected_images, projected_text = self.project_embeddings(
            image_embeddings,
            text_embedding,
        )

        # Postojeći Retriever pronalazi top-k slika na osnovu
        # cosine similarity vrednosti.
        indices, scores = self.retriever.retrieve_image(
            projected_text.squeeze(0),
            projected_images,
            top_k=top_k,
        )

        return indices, scores
