import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Dataset nad prethodno generisanim image i text embeddingima."""

    def __init__(self, embeddings: dict) -> None:
        self.sample_ids = embeddings["sample_ids"]
        self.image_embeddings = embeddings["image_embeddings"]
        self.text_embeddings = embeddings["text_embeddings"]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict:
        return {
            "sample_id": self.sample_ids[index],
            "image_embeddings": self.image_embeddings[index],
            "text_embeddings": self.text_embeddings[index],
        }
