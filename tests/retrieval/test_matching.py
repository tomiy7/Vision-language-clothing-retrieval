import torch

from vision_language_clothing_retrieval.retrieval.matching import (
    similarity_matrix,
)


def test_similarity_matrix_shape():
    image_embeddings = torch.randn(4, 256)
    text_embeddings = torch.randn(6, 256)

    result = similarity_matrix(
        image_embeddings,
        text_embeddings,
    )

    assert result.shape == (4, 6)


def test_similarity_matrix_identical_embeddings():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = similarity_matrix(
        embeddings,
        embeddings,
    )

    expected = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    assert torch.allclose(result, expected)


def test_similarity_matrix_prefers_matching_embedding():
    image_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    text_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = similarity_matrix(
        image_embeddings,
        text_embeddings,
    )

    assert result.argmax(dim=1).tolist() == [0, 1]
