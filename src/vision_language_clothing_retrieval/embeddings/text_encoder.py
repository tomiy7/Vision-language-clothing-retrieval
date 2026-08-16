from abc import ABC, abstractmethod

import torch
from transformers import AutoModel, AutoTokenizer


class TextEncoder(ABC):
    """Interfejs za pretvaranje tekstualnog opisa u vektorsku reprezentaciju."""
    @abstractmethod
    def encode(self, text: str) -> list[float]:
        raise NotImplementedError


class DistilBERTTextEncoder(TextEncoder):
    """Text encoder zasnovan na pretrained DistilBERT modelu."""
    def __init__(self) -> None:
        # Tokenizer pretvara tekst u tokene koje DistilBERT može da obradi.
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # Učitava se pretrained DistilBERT model koji se koristi
        # za izdvajanje tekstualnih reprezentacija.
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.model.eval()

    def encode(self, text: str) -> list[float]:
        # Tekst se tokenizuje i pretvara u PyTorch tenzore.
        inputs = self.tokenizer(text, return_tensors="pt")

        # Model se koristi samo za inference, pa nisu potrebni gradijenti.
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Kao reprezentacija celog teksta koristi se izlaz prvog tokena.
        embedding = outputs.last_hidden_state[:, 0, :]

        # Uklanja se batch dimenzija i embedding se pretvara u Python listu.
        return embedding.squeeze(0).tolist()
