import torch

from vision_language_clothing_retrieval.retrieval.projection import (
    ImageProjection,
    TextProjection,
    MultimodalProjection
)


def test_image_projection_output_shape():
    projection = ImageProjection()

    embeddings = torch.randn(8, 512)

    result = projection(embeddings)

    assert result.shape == (8, 256)


def test_text_projection_output_shape():
    projection = TextProjection()

    embeddings = torch.randn(8, 768)

    result = projection(embeddings)

    assert result.shape == (8, 256)

def test_multimodal_projection_output_shapes():
    projection = MultimodalProjection()

    image_embeddings = torch.randn(8, 512)
    text_embeddings = torch.randn(8, 768)

    image_result, text_result = projection(
        image_embeddings,
        text_embeddings,
    )

    assert image_result.shape == (8, 256)
    assert text_result.shape == (8, 256)


def test_multimodal_projection_supports_gradients():
    projection = MultimodalProjection()

    image_embeddings = torch.randn(
        8,
        512,
        requires_grad=True,
    )
    text_embeddings = torch.randn(
        8,
        768,
        requires_grad=True,
    )

    image_result, text_result = projection(
        image_embeddings,
        text_embeddings,
    )

    loss = image_result.mean() + text_result.mean()

    loss.backward()

    assert image_embeddings.grad is not None
    assert text_embeddings.grad is not None
