import torch

from vision_language_clothing_retrieval.evaluation.image_to_text.evaluator import (
    ImageToTextEvaluator,
)


class IdentityRetriever:
    def project_embeddings(self, image_embeddings, text_embeddings):
        return image_embeddings, text_embeddings


def test_perfect_matches_give_recall_1():
    torch.manual_seed(0)
    shared_embeddings = torch.randn(6, 8)

    evaluator = ImageToTextEvaluator(
        retriever=IdentityRetriever(),
        image_embeddings=shared_embeddings,
        text_embeddings=shared_embeddings,
        sample_ids=[
            "item-id_0-image",
            "item-id_1-image",
            "item-id_2-image",
            "item-id_3-image",
            "item-id_4-image",
            "item-id_5-image",
        ],
    )

    results = evaluator.evaluate(k_values=(1, 5))

    assert results["recall@1"] == 1.0
    assert results["mrr"] == 1.0
    assert results["mean_rank"] == 1.0


def test_deterministic_worst_case_gives_last_rank():
    n = 3
    image_embeddings = torch.eye(n)
    text_embeddings = -torch.eye(n)

    evaluator = ImageToTextEvaluator(
        retriever=IdentityRetriever(),
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        sample_ids=[
            "item-id_0-image",
            "item-id_1-image",
            "item-id_2-image",
        ],
    )
    results = evaluator.evaluate(k_values=(1,))

    assert results["recall@1"] == 0.0
    assert results["mean_rank"] == float(n)
    assert results["mrr"] == 1.0 / n

