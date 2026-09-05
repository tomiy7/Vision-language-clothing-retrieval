import torch

from vision_language_clothing_retrieval.evaluation.text_to_image.evaluator import (
    TextToImageEvaluator,
)


class IdentityRetriever:
    """Retriever koji za svaki query vraća odgovarajuću sliku na prvom mestu."""

    def retrieve_image(
        self,
        text_embedding,
        image_embeddings,
        top_k=5,
    ):
        query_index = int(text_embedding.item())

        indices = torch.cat(
            (
                torch.tensor([query_index]),
                torch.tensor(
                    [
                        i
                        for i in range(len(image_embeddings))
                        if i != query_index
                    ]
                ),
            )
        )[:top_k]

        scores = torch.arange(
            len(indices),
            0,
            -1,
            dtype=torch.float,
        )

        return indices, scores


class WorstCaseRetriever:
    """Retriever koji za svaki query stavlja tačnu sliku na poslednje mesto."""

    def retrieve_image(
        self,
        text_embedding,
        image_embeddings,
        top_k=5,
    ):
        query_index = int(text_embedding.item())

        indices = torch.tensor(
            [
                i
                for i in range(len(image_embeddings))
                if i != query_index
            ]
            + [query_index]
        )[:top_k]

        scores = torch.arange(
            len(indices),
            0,
            -1,
            dtype=torch.float,
        )

        return indices, scores


def test_text_to_image_perfect_matches_give_recall_1():
    text_embeddings = torch.arange(6, dtype=torch.float).unsqueeze(1)
    image_embeddings = torch.randn(6, 8)

    evaluator = TextToImageEvaluator(
        retriever=IdentityRetriever(),
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        sample_ids=[
            "item-id_0-image",
            "item-id_1-image",
            "item-id_2-image",
            "item-id_3-image",
            "item-id_4-image",
            "item-id_5-image",
        ],
    )

    results = evaluator.evaluate()

    assert results["recall@1"] == 1.0
    assert results["recall@5"] == 1.0
    assert results["recall@10"] == 1.0
    assert results["mrr"] == 1.0
    assert results["mean_rank"] == 1.0


def test_text_to_image_worst_case_gives_low_recall():
    text_embeddings = torch.arange(6, dtype=torch.float).unsqueeze(1)
    image_embeddings = torch.randn(6, 8)

    evaluator = TextToImageEvaluator(
        retriever=WorstCaseRetriever(),
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        sample_ids=[
            "item-id_0-image",
            "item-id_1-image",
            "item-id_2-image",
            "item-id_3-image",
            "item-id_4-image",
            "item-id_5-image",
        ],
    )
    results = evaluator.evaluate()

    assert results["recall@1"] == 0.0
    assert results["recall@5"] == 0.0
    assert results["recall@10"] == 1.0
    assert results["mean_rank"] == 6.0