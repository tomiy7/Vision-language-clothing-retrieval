from abc import ABC, abstractmethod

import torch
from transformers import AutoModel, AutoTokenizer


class TextEncoder(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[float]:
        """Convert text into a vector representation."""
        raise NotImplementedError


class DistilBERTTextEncoder(TextEncoder):
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.model.eval()

    def encode(self, text: str) -> list[float]:
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :]

        return embedding.squeeze(0).tolist()
