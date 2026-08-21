import torch
import torch.nn.functional as F


class Retriever:
    """Pronalaženje najsličnijih image/text embeddinga."""

    def retrieve_text(
        self,
        image_embedding: torch.Tensor,
        text_embeddings: torch.Tensor,
        top_k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Za dati image embedding pronalazi top-k najsličnijih text embeddinga.

        Vraća:
            indices: indekse pronađenih tekstova
            scores: njihove cosine similarity vrednosti
        """

        # Normalizacija omogućava da skalarni proizvod odgovara
        # cosine similarity vrednosti.
        image_embedding = F.normalize(
            image_embedding,
            p=2,
            dim=-1,
        )

        text_embeddings = F.normalize(
            text_embeddings,
            p=2,
            dim=-1,
        )

        # Računa se sličnost datog image embeddinga sa svim
        # dostupnim text embeddingima.
        similarities = text_embeddings @ image_embedding

        # Biraju se top-k tekstova sa najvećom similarity vrednošću.
        scores, indices = torch.topk(
            similarities,
            k=top_k,
        )

        return indices, scores

    def retrieve_image(
        self,
        text_embedding: torch.Tensor,
        image_embeddings: torch.Tensor,
        top_k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Za dati text embedding pronalazi top-k najsličnijih image embeddinga.
        """

        # Normalizacija omogućava da skalarni proizvod odgovara
        # cosine similarity vrednosti.
        text_embedding = F.normalize(
            text_embedding,
            p=2,
            dim=-1,
        )

        image_embeddings = F.normalize(
            image_embeddings,
            p=2,
            dim=-1,
        )

        # Računa se sličnost datog text embeddinga sa svim
        # dostupnim image embeddingima.
        similarities = image_embeddings @ text_embedding

        # Biraju se top-k slika sa najvećom similarity vrednošću.
        scores, indices = torch.topk(
            similarities,
            k=top_k,
        )

        return indices, scores
