import torch
import torch.nn.functional as F


def similarity_matrix(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Računa cosine similarity između svih image i text embeddinga."""

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

    # Računa se similarity između svakog image i svakog text embeddinga.
    # Dobija se matrica oblika [broj_slika, broj_tekstova].
    return image_embeddings @ text_embeddings.T
