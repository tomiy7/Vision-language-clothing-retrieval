import torch
import torch.nn.functional as F


def contrastive_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Računa kontrastivni loss za image-text embeddinge."""

    # Normalizuju se image i text embeddingi kako bi njihov
    # skalarni proizvod odgovarao cosine similarity vrednosti.
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
    # Rezultat je matrica oblika [batch_size, batch_size].
    logits = image_embeddings @ text_embeddings.T

    # Temperature kontroliše koliko su similarity vrednosti "oštre"
    # prilikom računanja softmax distribucije u cross-entropy loss-u.
    logits = logits / temperature

    # Za svaki image embedding odgovarajući text embedding nalazi se
    # na istoj poziciji u batch-u, pa se ta pozicija koristi kao label.
    labels = torch.arange(
        image_embeddings.size(0),
        device=image_embeddings.device,
    )

    # Image-to-text loss: za svaki image model treba da prepozna
    # odgovarajući text embedding među svim tekstovima u batch-u.
    image_to_text_loss = F.cross_entropy(
        logits,
        labels,
    )

    # Text-to-image loss: isti princip se primenjuje u suprotnom smeru.
    text_to_image_loss = F.cross_entropy(
        logits.T,
        labels,
    )

    # Konačni kontrastivni loss predstavlja prosek oba pravca,
    # čime se istovremeno uči image-to-text i text-to-image matching.
    return (
        image_to_text_loss + text_to_image_loss
    ) / 2
