from pathlib import Path

import torch


def save_embeddings(
    embeddings: dict,
    path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, path)


def load_embeddings(path: str) -> dict:
    return torch.load(path, weights_only=True)
