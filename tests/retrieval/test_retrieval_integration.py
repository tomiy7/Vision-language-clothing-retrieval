import torch

from vision_language_clothing_retrieval.retrieval.multimodal_retriever import (
    MultimodalRetriever,
)


def test_retrieval_end_to_end():
    embeddings = torch.load(
        "embeddings/train.pt",
        weights_only=True,
    )

    retriever = MultimodalRetriever(
        model_path="embeddings/multimodal_model.pt",
    )

    image_embeddings = embeddings["image_embeddings"]
    text_embeddings = embeddings["text_embeddings"]

    image_embedding = image_embeddings[0]
    text_embedding = text_embeddings[0]

    text_indices, text_scores = retriever.retrieve_text(
        image_embedding,
        text_embeddings,
        top_k=5,
    )

    image_indices, image_scores = retriever.retrieve_image(
        text_embedding,
        image_embeddings,
        top_k=5,
    )

    assert text_indices.shape == (5,)
    assert text_scores.shape == (5,)

    assert image_indices.shape == (5,)
    assert image_scores.shape == (5,)
