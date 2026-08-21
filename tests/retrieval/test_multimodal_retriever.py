import torch

from vision_language_clothing_retrieval.retrieval.multimodal_retriever import (
    MultimodalRetriever,
)


def test_project_embeddings_output_shapes(tmp_path):
    retriever = MultimodalRetriever(
        model_path="embeddings/multimodal_model.pt",
    )

    image_embeddings = torch.randn(8, 512)
    text_embeddings = torch.randn(8, 768)

    projected_images, projected_texts = retriever.project_embeddings(
        image_embeddings,
        text_embeddings,
    )

    assert projected_images.shape == (8, 256)
    assert projected_texts.shape == (8, 256)


def test_retrieve_text_returns_top_k():
    retriever = MultimodalRetriever(
        model_path="embeddings/multimodal_model.pt",
    )

    image_embedding = torch.randn(1, 512)
    text_embeddings = torch.randn(10, 768)

    indices, scores = retriever.retrieve_text(
        image_embedding,
        text_embeddings,
        top_k=3,
    )

    assert indices.shape == (3,)
    assert scores.shape == (3,)

def test_retrieve_image_returns_top_k():
    retriever = MultimodalRetriever(
        model_path="embeddings/multimodal_model.pt",
    )

    text_embedding = torch.randn(1, 768)
    image_embeddings = torch.randn(10, 512)

    indices, scores = retriever.retrieve_image(
        text_embedding,
        image_embeddings,
        top_k=3,
    )

    assert indices.shape == (3,)
    assert scores.shape == (3,)
