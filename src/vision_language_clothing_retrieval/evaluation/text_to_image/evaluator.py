import torch

from vision_language_clothing_retrieval.evaluation.metrics import (
    mean_rank,
    mean_reciprocal_rank,
    recall_at_k,
)
from vision_language_clothing_retrieval.retrieval.multimodal_retriever import (
    MultimodalRetriever,
)


class TextToImageEvaluator:
    def __init__(
        self,
        retriever: MultimodalRetriever,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> None:
        self.retriever = retriever
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings

    def get_rank(
        self,
        text_embedding: torch.Tensor,
        correct_index: int,
    ) -> int:
        """Pronalazi rank odgovarajuće slike za dati text query."""

        indices, _ = self.retriever.retrieve_image(
            text_embedding=text_embedding.unsqueeze(0),
            image_embeddings=self.image_embeddings,
            top_k=len(self.image_embeddings),
        )

        matches = (indices == correct_index).nonzero(as_tuple=True)[0]

        if len(matches) == 0:
            raise ValueError(
                f"Correct image with index {correct_index} was not found."
            )

        return matches[0].item() + 1

    def evaluate(self) -> dict[str, float]:
        """Evaluira Text → Image retrieval nad celim skupu."""

        ranks = []

        for index, text_embedding in enumerate(self.text_embeddings):
            rank = self.get_rank(
                text_embedding=text_embedding,
                correct_index=index,
            )

            ranks.append(rank)

        return {
            "recall@1": recall_at_k(ranks, 1),
            "recall@5": recall_at_k(ranks, 5),
            "recall@10": recall_at_k(ranks, 10),
            "mrr": mean_reciprocal_rank(ranks),
            "mean_rank": mean_rank(ranks),
        }