import io

import torch
from fastapi.testclient import TestClient

from vision_language_clothing_retrieval.api.main import app
from vision_language_clothing_retrieval.api.routes import get_retrieval_service


class FakeRetrievalService:
    """Fake verzija RetrievalService - isti interfejs, bez pravog
    modela/embeddinga/enkodera."""

    def retrieve_images(self, text: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k mora biti veći od 0.")

        return [
            {
                "sample_id": f"sample_{i}",
                "image_path": f"data/images/sample_{i}.jpg",
                "text": f"caption for sample_{i}",
                "score": round(1.0 - i * 0.1, 4),
            }
            for i in range(top_k)
        ]

    def retrieve_texts(self, image_path: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k mora biti veći od 0.")

        return [
            {
                "sample_id": f"sample_{i}",
                "image_path": f"data/images/sample_{i}.jpg",
                "text": f"caption for sample_{i}",
                "score": round(1.0 - i * 0.1, 4),
            }
            for i in range(top_k)
        ]


def _client_with_fake_service() -> TestClient:
    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()
    return TestClient(app)


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_by_text_returns_image_results():
    client = _client_with_fake_service()

    response = client.post(
        "/search",
        data={"query_type": "text", "text": "red dress", "top_k": "3"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["query_type"] == "text"
    assert len(data["results"]) == 3
    assert all(r["result_type"] == "image" for r in data["results"])


def test_search_by_text_missing_text_field_fails():
    client = _client_with_fake_service()

    response = client.post(
        "/search",
        data={"query_type": "text"},
    )

    assert response.status_code == 422


def test_search_by_image_returns_text_results():
    client = _client_with_fake_service()

    fake_image = io.BytesIO(b"fake image bytes")

    response = client.post(
        "/search",
        data={"query_type": "image", "top_k": "2"},
        files={"image": ("query.jpg", fake_image, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["query_type"] == "image"
    assert len(data["results"]) == 2
    assert all(r["result_type"] == "text" for r in data["results"])


def test_search_by_image_missing_file_fails():
    client = _client_with_fake_service()

    response = client.post(
        "/search",
        data={"query_type": "image"},
    )

    assert response.status_code == 422

def test_search_invalid_query_type_fails():
    client = _client_with_fake_service()

    response = client.post(
        "/search",
        data={"query_type": "video", "text": "red dress"},
    )

    assert response.status_code == 422

def test_search_missing_query_type_fails():
    client = _client_with_fake_service()

    response = client.post(
        "/search",
        data={"text": "red dress"},
    )

    assert response.status_code == 422

def test_search_by_text_uses_default_top_k_when_not_provided():
    client = _client_with_fake_service()

    response = client.post(
        "/search",
        data={"query_type": "text", "text": "red dress"},
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 5

def test_search_by_text_empty_string_fails_same_as_missing():
    client = _client_with_fake_service()

    response = client.post(
        "/search",
        data={"query_type": "text", "text": ""},
    )

    assert response.status_code == 422

def test_search_by_image_without_filename_is_rejected():
    client = _client_with_fake_service()

    fake_image = io.BytesIO(b"fake image bytes")

    response = client.post(
        "/search",
        data={"query_type": "image", "top_k": "2"},
        files={"image": ("", fake_image, "image/jpeg")},
    )

    assert response.status_code == 422

def test_search_top_k_zero_returns_422():
    client = _client_with_fake_service()

    response = client.post(
        "/search",
        data={"query_type": "text", "text": "red dress", "top_k": "0"},
    )

    assert response.status_code == 422
