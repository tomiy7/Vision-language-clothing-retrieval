import torch

from vision_language_clothing_retrieval.retrieval.loss import (
    contrastive_loss,
)


def test_contrastive_loss_returns_scalar():
    image_embeddings = torch.randn(4, 256)
    text_embeddings = torch.randn(4, 256)

    loss = contrastive_loss(
        image_embeddings,
        text_embeddings,
    )

    assert loss.ndim == 0


def test_contrastive_loss_is_positive():
    image_embeddings = torch.randn(4, 256)
    text_embeddings = torch.randn(4, 256)

    loss = contrastive_loss(
        image_embeddings,
        text_embeddings,
    )

    assert loss.item() > 0


def test_contrastive_loss_supports_gradients():
    image_embeddings = torch.randn(
        4,
        256,
        requires_grad=True,
    )
    text_embeddings = torch.randn(
        4,
        256,
        requires_grad=True,
    )

    loss = contrastive_loss(
        image_embeddings,
        text_embeddings,
    )

    loss.backward()

    assert image_embeddings.grad is not None
    assert text_embeddings.grad is not None

def test_contrastive_loss_prefers_matching_embeddings():
    image_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    matching_text_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    mismatched_text_embeddings = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    matching_loss = contrastive_loss(
        image_embeddings,
        matching_text_embeddings,
    )

    mismatched_loss = contrastive_loss(
        image_embeddings,
        mismatched_text_embeddings,
    )

    assert matching_loss < mismatched_loss
