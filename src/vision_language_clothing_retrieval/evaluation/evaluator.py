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