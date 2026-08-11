from pathlib import Path

import torch


def save_embeddings(
    embeddings: dict,
    path: str,
) -> None:
    """Čuva generisane embeddinge u PyTorch fajl."""

    # Kreira se direktorijum za izlazni fajl ako već ne postoji.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, path)


def load_embeddings(path: str) -> dict:
    """Učitava prethodno sačuvane embeddinge iz PyTorch fajla."""

    return torch.load(path, weights_only=True)
