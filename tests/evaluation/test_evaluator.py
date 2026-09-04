import torch

from vision_language_clothing_retrieval.evaluation.evaluator import (
    ImageToTextEvaluator,
    TextToImageEvaluator
)

class IdentityRetriever:
    def project_embeddings(self, image_embeddings, text_embeddings):
        return image_embeddings, text_embeddings


def test_image_to_text_perfect_matches_give_recall_1():
    torch.manual_seed(0)
    shared_embeddings = torch.randn(6, 8)

    evaluator = ImageToTextEvaluator(
        retriever = IdentityRetriever(),
        image_embeddings=shared_embeddings,
        text_embeddings=shared_embeddings,
    )

    results = evaluator.evaluate(k_values=(1, 3, 5))

    assert results["recall@1"] == 1.0
    assert results["mean_reciprocal_rank"] == 1.0
    assert results["mean_rank"] == 1.0

def test_image_to_text_worst_case_gives_low_recall():
    torch.manual_seed(0)
    image_embeddings = torch.randn(6, 8)

    text_embeddings = torch.roll(image_embeddings, 1, 0)

    evaluator = ImageToTextEvaluator(
        retriever = IdentityRetriever(),
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
    )

    results = evaluator.evaluate(k_values=(1, 3, 5))

    assert results["recall@1"] == 0.0
    assert results["mean_rank"] > 1.0