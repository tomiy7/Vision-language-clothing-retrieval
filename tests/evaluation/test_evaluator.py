import torch

from vision_language_clothing_retrieval.evaluation.text_to_image.evaluator import (
    TextToImageEvaluator,
)

class FakeRetriever:
    """Fake retriever koji vraća unapred definisane rezultate."""

    def retrieve_image(
        self,
        text_embedding: torch.Tensor,
        image_embeddings: torch.Tensor,
        top_k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        query_index = int(text_embedding.squeeze().item())

        results = {
            0: [0, 1, 2],
            1: [2, 1, 0],
            2: [1, 0, 2],
        }

        indices = torch.tensor(results[query_index])

        scores = torch.tensor([0.9, 0.8, 0.7])

        return indices, scores


def create_evaluator() -> TextToImageEvaluator:
    """Kreira evaluator sa malim test skupom."""

    retriever = FakeRetriever()

    image_embeddings = torch.tensor(
        [
            [10.0],
            [20.0],
            [30.0],
        ]
    )

    text_embeddings = torch.tensor(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    return TextToImageEvaluator(
        retriever=retriever,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        sample_ids=[
            "item-id_0-image",
            "item-id_1-image",
            "item-id_2-image",
        ],
    )


def test_get_rank():
    evaluator = create_evaluator()

    rank = evaluator.get_rank(
        text_embedding=evaluator.text_embeddings[1],
        correct_sample_id=evaluator.sample_ids[1],
    )

    assert rank == 2


def test_evaluate():
    evaluator = create_evaluator()

    results = evaluator.evaluate()

    assert results["recall@1"] == 1 / 3
    assert results["recall@5"] == 1.0
    assert results["recall@10"] == 1.0

    expected_mrr = (1 / 1 + 1 / 2 + 1 / 3) / 3

    assert results["mrr"] == expected_mrr
    assert results["mean_rank"] == 2.0