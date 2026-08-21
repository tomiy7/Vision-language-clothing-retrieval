import torch
import torch.nn.functional as F


def cosine_similarity(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Računa cosine similarity između image i text embeddinga."""

    # Normalizuju se embeddingi kako bi skalarni proizvod
    # odgovarao cosine similarity vrednosti.
    image_embeddings = F.normalize(
        image_embeddings,
        p=2,
        dim=-1,
    )
    text_embeddings = F.normalize(
        text_embeddings,
        p=2,
        dim=-1,
    )

    # Skalarni proizvod normalizovanih embeddinga daje
    # cosine similarity za odgovarajuće parove.
    return torch.sum(
        image_embeddings * text_embeddings,
        dim=-1,
    )
