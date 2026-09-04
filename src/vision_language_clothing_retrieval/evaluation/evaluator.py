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


def _ranks_from_similarity(similarity_matrix: torch.Tensor) -> list[int]:
    n_queries = similarity_matrix.shape[0]
    ranks = []

    for i in range(n_queries):
        scores = similarity_matrix[i]
        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    return ranks

def _metrics_from_ranks(ranks: list[int], k_values: tuple[int, ...]) -> dict:
    results = {
        "mean_rank": mean_rank(ranks),
        "mean_reciprocal_rank": mean_reciprocal_rank(ranks),
    }
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(ranks, k)
    return results


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


class ImageToTextEvaluator:
    def __init__(
            self,
            retriever: MultimodalRetriever,
            image_embeddings: torch.Tensor,
            text_embeddings: torch.Tensor,
    ) -> None:
        self.retriever = retriever
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings

    def evaluate(self, k_values: tuple[int, ...] = (1, 5, 10)) -> dict:
        projected_images, projected_texts = self.retriever.project_embeddings(
            self.image_embeddings,
            self.text_embeddings
        )

        projected_images = F.normalize(projected_images, p=1, dim=-1)
        projected_texts = F.normalize(projected_texts, p=1, dim=-1)

        similarity_matrix = projected_images @ projected_texts.T;

        ranks = _ranks_from_similarity(similarity_matrix)
        return _metrics_from_ranks(ranks, k_values)