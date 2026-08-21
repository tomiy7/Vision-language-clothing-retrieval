import torch

from vision_language_clothing_retrieval.retrieval.retriever import (
    Retriever,
)


def test_retrieve_text_returns_top_k():
    retriever = Retriever()

    image_embedding = torch.tensor(
        [1.0, 0.0, 0.0]
    )

    text_embeddings = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
        ]
    )

    indices, scores = retriever.retrieve_text(
        image_embedding,
        text_embeddings,
        top_k=2,
    )

    assert indices.tolist() == [1, 3]

    assert torch.allclose(
        scores,
        torch.tensor([1.0, 0.8]),
        atol=1e-6,
    )

def test_retrieve_image_returns_top_k():
    retriever = Retriever()

    text_embedding = torch.tensor(
        [1.0, 0.0, 0.0]
    )

    image_embeddings = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
        ]
    )

    indices, scores = retriever.retrieve_image(
        text_embedding,
        image_embeddings,
        top_k=2,
    )

    assert indices.tolist() == [1, 3]

    assert torch.allclose(
        scores,
        torch.tensor([1.0, 0.8]),
        atol=1e-6,
    )


def test_retrieve_text_respects_top_k():
    retriever = Retriever()

    image_embedding = torch.tensor(
        [1.0, 0.0, 0.0]
    )

    text_embeddings = torch.randn(10, 3)

    indices, scores = retriever.retrieve_text(
        image_embedding,
        text_embeddings,
        top_k=3,
    )

    assert indices.shape == (3,)
    assert scores.shape == (3,)
