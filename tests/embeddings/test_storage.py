import torch

from vision_language_clothing_retrieval.embeddings.storage import (
    load_embeddings,
    save_embeddings,
)


def test_save_and_load_embeddings(tmp_path):
    embeddings = {
        "sample_ids": ["sample_1", "sample_2"],
        "image_embeddings": torch.randn(2, 512),
        "text_embeddings": torch.randn(2, 768),
    }

    path = tmp_path / "embeddings.pt"

    save_embeddings(embeddings, str(path))
    loaded = load_embeddings(str(path))

    assert loaded["sample_ids"] == embeddings["sample_ids"]
    assert torch.equal(
        loaded["image_embeddings"],
        embeddings["image_embeddings"],
    )
    assert torch.equal(
        loaded["text_embeddings"],
        embeddings["text_embeddings"],
    )
