from unittest.mock import MagicMock, patch

import pytest
import torch

from vision_language_clothing_retrieval.dataset.sample import DatasetSample
from vision_language_clothing_retrieval.services.retrieval_service import (
    RetrievalService,
)


def create_service():
    retriever = MagicMock()
    text_encoder = MagicMock()
    image_encoder = MagicMock()
    loader = MagicMock()

    samples = [
        DatasetSample(
            sample_id="image_0",
            image_path="images/image_0.jpg",
            text="black dress",
        ),
        DatasetSample(
            sample_id="image_1",
            image_path="images/image_1.jpg",
            text="white shirt",
        ),
        DatasetSample(
            sample_id="image_2",
            image_path="images/image_2.jpg",
            text="blue jeans",
        ),
    ]

    loader.load.return_value = samples

    patches = [
        patch(
            "vision_language_clothing_retrieval.services.retrieval_service.Path.exists",
            return_value=True,
        ),
        patch(
            "vision_language_clothing_retrieval.services.retrieval_service.MultimodalRetriever",
            return_value=retriever,
        ),
        patch(
            "vision_language_clothing_retrieval.services.retrieval_service.DistilBERTTextEncoder",
            return_value=text_encoder,
        ),
        patch(
            "vision_language_clothing_retrieval.services.retrieval_service.ResNet10ImageEncoder",
            return_value=image_encoder,
        ),
        patch(
            "vision_language_clothing_retrieval.services.retrieval_service.DeepFashionDatasetLoader",
            return_value=loader,
        ),
        patch(
            "vision_language_clothing_retrieval.services.retrieval_service.torch.load",
            return_value={
                "sample_ids": [
                    "image_0",
                    "image_1",
                    "image_2",
                ],
                "image_embeddings": torch.randn(3, 512),
                "text_embeddings": torch.randn(3, 768),
            },
        ),
    ]

    started_patches = [patcher.start() for patcher in patches]

    service = RetrievalService(
        model_path="model.pt",
        embeddings_path="embeddings.pt",
        image_dir="images",
        captions_path="captions.json",
    )

    return (
        service,
        retriever,
        text_encoder,
        image_encoder,
        loader,
        patches,
        started_patches,
    )


def stop_patches(patches):
    for patcher in patches:
        patcher.stop()


def test_retrieval_service_loads_model_embeddings_and_samples():
    (
        service,
        retriever,
        text_encoder,
        image_encoder,
        loader,
        patches,
        started_patches,
    ) = create_service()

    try:
        assert service.image_embeddings.shape == (3, 512)
        assert service.text_embeddings.shape == (3, 768)

        assert service.sample_ids == [
            "image_0",
            "image_1",
            "image_2",
        ]

        assert service.samples["image_0"].text == "black dress"
        assert service.samples["image_1"].image_path == "images/image_1.jpg"
    finally:
        stop_patches(patches)


def test_retrieve_images_creates_text_embedding_and_formats_results():
    (
        service,
        retriever,
        text_encoder,
        image_encoder,
        loader,
        patches,
        started_patches,
    ) = create_service()

    try:
        text_encoder.encode.return_value = [0.1] * 768

        retriever.retrieve_image.return_value = (
            torch.tensor([1, 2]),
            torch.tensor([0.95, 0.85]),
        )

        results = service.retrieve_images(
            text="white shirt",
            top_k=2,
        )

        text_encoder.encode.assert_called_once_with("white shirt")

        call = retriever.retrieve_image.call_args

        actual_text_embedding = call.kwargs["text_embedding"]

        assert torch.equal(
            actual_text_embedding,
            torch.tensor([0.1] * 768),
        )

        assert torch.equal(
            call.kwargs["image_embeddings"],
            service.image_embeddings,
        )

        assert call.kwargs["top_k"] == 2

        assert results[0]["sample_id"] == "image_1"
        assert results[0]["image_path"] == "images/image_1.jpg"
        assert results[0]["text"] == "white shirt"
        assert results[0]["score"] == pytest.approx(0.95)

        assert results[1]["sample_id"] == "image_2"
        assert results[1]["image_path"] == "images/image_2.jpg"
        assert results[1]["text"] == "blue jeans"
        assert results[1]["score"] == pytest.approx(0.85)

    finally:
        stop_patches(patches)


def test_retrieve_texts_creates_image_embedding_and_formats_results():
    (
        service,
        retriever,
        text_encoder,
        image_encoder,
        loader,
        patches,
        started_patches,
    ) = create_service()

    try:
        image_encoder.encode.return_value = [0.2] * 512

        retriever.retrieve_text.return_value = (
            torch.tensor([2, 0]),
            torch.tensor([0.93, 0.81]),
        )

        results = service.retrieve_texts(
            image_path="query.jpg",
            top_k=2,
        )

        image_encoder.encode.assert_called_once_with("query.jpg")

        call = retriever.retrieve_text.call_args

        actual_image_embedding = call.kwargs["image_embedding"]

        assert torch.equal(
            actual_image_embedding,
            torch.tensor([0.2] * 512),
        )

        assert torch.equal(
            call.kwargs["text_embeddings"],
            service.text_embeddings,
        )

        assert call.kwargs["top_k"] == 2

        assert results[0]["sample_id"] == "image_2"
        assert results[0]["image_path"] == "images/image_2.jpg"
        assert results[0]["text"] == "blue jeans"
        assert results[0]["score"] == pytest.approx(0.93)

        assert results[1]["sample_id"] == "image_0"
        assert results[1]["image_path"] == "images/image_0.jpg"
        assert results[1]["text"] == "black dress"
        assert results[1]["score"] == pytest.approx(0.81)

    finally:
        stop_patches(patches)


def test_retrieve_images_rejects_invalid_top_k():
    (
        service,
        retriever,
        text_encoder,
        image_encoder,
        loader,
        patches,
        started_patches,
    ) = create_service()

    try:
        try:
            service.retrieve_images(
                text="black dress",
                top_k=0,
            )
            assert False, "Očekivan je ValueError."
        except ValueError as error:
            assert str(error) == "top_k mora biti veći od 0."
    finally:
        stop_patches(patches)


def test_retrieve_texts_rejects_invalid_top_k():
    (
        service,
        retriever,
        text_encoder,
        image_encoder,
        loader,
        patches,
        started_patches,
    ) = create_service()

    try:
        try:
            service.retrieve_texts(
                image_path="query.jpg",
                top_k=0,
            )
            assert False, "Očekivan je ValueError."
        except ValueError as error:
            assert str(error) == "top_k mora biti veći od 0."
    finally:
        stop_patches(patches)