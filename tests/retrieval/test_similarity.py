import torch

from vision_language_clothing_retrieval.retrieval.similarity import (
    cosine_similarity,
)


def test_cosine_similarity_identical_embeddings():
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    result = cosine_similarity(embeddings, embeddings)

    assert torch.allclose(result, torch.ones(2))


def test_cosine_similarity_orthogonal_embeddings():
    image_embeddings = torch.tensor(
        [[1.0, 0.0, 0.0]]
    )
    text_embeddings = torch.tensor(
        [[0.0, 1.0, 0.0]]
    )

    result = cosine_similarity(
        image_embeddings,
        text_embeddings,
    )

    assert torch.allclose(result, torch.zeros(1))


def test_cosine_similarity_opposite_embeddings():
    image_embeddings = torch.tensor(
        [[1.0, 0.0, 0.0]]
    )
    text_embeddings = torch.tensor(
        [[-1.0, 0.0, 0.0]]
    )

    result = cosine_similarity(
        image_embeddings,
        text_embeddings,
    )

    assert torch.allclose(result, torch.tensor([-1.0]))

def test_cosine_similarity_batch():
    image_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    text_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    result = cosine_similarity(
        image_embeddings,
        text_embeddings,
    )

    assert result.shape == (3,)
    assert torch.allclose(
        result,
        torch.tensor([1.0, 1.0, 0.7071]),
        atol=1e-4,
    )
