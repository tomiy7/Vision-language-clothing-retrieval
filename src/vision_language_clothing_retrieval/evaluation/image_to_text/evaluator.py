import torch
import torch.nn.functional as F

from vision_language_clothing_retrieval.evaluation.metrics import (
    mean_rank,
    mean_reciprocal_rank,
    recall_at_k,
)
from vision_language_clothing_retrieval.retrieval.multimodal_retriever import (
    MultimodalRetriever,
)

class ImageToTextEvaluator:
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

    def evaluate(self, k_values: tuple[int, ...] = (1, 5, 10)) -> dict:
        projected_images, projected_texts = self.retriever.project_embeddings(
            self.image_embeddings,
            self.text_embeddings
        )

        projected_images = F.normalize(projected_images, p=2, dim=-1)
        projected_texts = F.normalize(projected_texts, p=2, dim=-1)

        similarity_matrix = projected_images @ projected_texts.T

        ranks = self._ranks_from_similarity(
            similarity_matrix,
            self.sample_ids
        )

        results = {
            "mean_rank": mean_rank(ranks),
            "mrr": mean_reciprocal_rank(ranks),
        }
        for k in k_values:
            results[f"recall@{k}"] = recall_at_k(ranks, k)

        return results


    @staticmethod
    def _ranks_from_similarity(
            similarity_matrix: torch.Tensor,
            sample_ids: list[str],
    ) -> list[int]:
        n_queries = similarity_matrix.shape[0]
        ranks = []

        for i in range(n_queries):
            scores = similarity_matrix[i]
            sorted_indices = torch.argsort(scores, descending=True)

            correct_item_id = ImageToTextEvaluator._get_item_id(
                sample_ids[i]
            )

            for rank, index in enumerate(sorted_indices, start=1):
                retrieved_item_id = ImageToTextEvaluator._get_item_id(
                    sample_ids[index.item()]
                )

                if retrieved_item_id == correct_item_id:
                    ranks.append(rank)
                    break

        return ranks

    @staticmethod
    def _get_item_id(sample_id: str) -> str:

        return sample_id.split("-id_")[1].split("-")[0]