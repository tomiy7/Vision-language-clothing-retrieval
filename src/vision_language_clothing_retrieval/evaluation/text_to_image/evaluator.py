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
        sample_ids: list[str],
    ) -> None:
        self.retriever = retriever
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
        self.sample_ids = sample_ids

    def get_rank(
        self,
        text_embedding: torch.Tensor,
        correct_sample_id: str,
    ) -> int:
        """Pronalazi rank prve slike istog clothing item-a za dati text query."""

        indices, _ = self.retriever.retrieve_image(
            text_embedding=text_embedding.unsqueeze(0),
            image_embeddings=self.image_embeddings,
            top_k=len(self.image_embeddings),
        )

        correct_item_id = self._get_item_id(correct_sample_id)

        for rank, index in enumerate(indices, start=1):
            retrieved_sample_id = self.sample_ids[index.item()]
            retrieved_item_id = self._get_item_id(retrieved_sample_id)

            if retrieved_item_id == correct_item_id:
                return rank

        raise ValueError(
            f"Correct clothing item {correct_item_id} was not found."
        )

    def evaluate(self) -> dict[str, float]:
        """Evaluira Text → Image retrieval nad celim skupu."""

        ranks = []

        for index, text_embedding in enumerate(self.text_embeddings):
            rank = self.get_rank(
                text_embedding=text_embedding,
                correct_sample_id=self.sample_ids[index],
            )

            ranks.append(rank)

        return {
            "recall@1": recall_at_k(ranks, 1),
            "recall@5": recall_at_k(ranks, 5),
            "recall@10": recall_at_k(ranks, 10),
            "mrr": mean_reciprocal_rank(ranks),
            "mean_rank": mean_rank(ranks),
        }

    @staticmethod
    def _get_item_id(sample_id: str) -> str:
        return sample_id.split("-id_")[1].split("-")[0]