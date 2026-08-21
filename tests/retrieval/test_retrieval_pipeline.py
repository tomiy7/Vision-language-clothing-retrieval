import torch

from vision_language_clothing_retrieval.retrieval.projection import (
    MultimodalProjection,
)
from vision_language_clothing_retrieval.retrieval.retriever import (
    Retriever,
)


def test_image_to_text_retrieval_pipeline():
    projection = MultimodalProjection()
    retriever = Retriever()

    image_embeddings = torch.randn(4, 512)
    text_embeddings = torch.randn(4, 768)

    projected_images, projected_texts = projection(
        image_embeddings,
        text_embeddings,
    )

    indices, scores = retriever.retrieve_text(
        projected_images[0],
        projected_texts,
        top_k=2,
    )

    assert projected_images.shape == (4, 256)
    assert projected_texts.shape == (4, 256)

    assert indices.shape == (2,)
    assert scores.shape == (2,)

    assert torch.isfinite(scores).all()

def test_text_to_image_retrieval_pipeline():
    projection = MultimodalProjection()
    retriever = Retriever()

    image_embeddings = torch.randn(4, 512)
    text_embeddings = torch.randn(4, 768)

    projected_images, projected_texts = projection(
        image_embeddings,
        text_embeddings,
    )

    indices, scores = retriever.retrieve_image(
        projected_texts[0],
        projected_images,
        top_k=2,
    )

    assert projected_images.shape == (4, 256)
    assert projected_texts.shape == (4, 256)

    assert indices.shape == (2,)
    assert scores.shape == (2,)

    assert torch.isfinite(scores).all()
