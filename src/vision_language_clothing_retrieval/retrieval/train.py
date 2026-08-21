from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vision_language_clothing_retrieval.retrieval.embedding_dataset import (
    EmbeddingDataset,
)
from vision_language_clothing_retrieval.retrieval.model import (
    MultimodalModel,
)
from vision_language_clothing_retrieval.retrieval.trainer import (
    MultimodalTrainer,
)


def load_embeddings(path: str) -> dict:
    """Učitava prethodno generisane embeddinge."""

    # Embeddingi se učitavaju iz prethodno sačuvanog PyTorch fajla.
    return torch.load(
        path,
        weights_only=True,
    )


def train(
    embeddings_path: str,
    output_path: str,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    temperature: float = 0.07,
    device: str = "cpu",
) -> None:
    """Trenira multimodalni model nad postojećim embeddingima."""

    # Učitavaju se prethodno generisani image i text embeddingi.
    embeddings = load_embeddings(embeddings_path)

    # Od učitanih embeddinga formira se Dataset koji omogućava
    # njihovu obradu u batch-evima.
    dataset = EmbeddingDataset(embeddings)

    # DataLoader organizuje podatke u batch-eve i meša njihov redosled
    # između epoha.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Kreira se multimodalni model koji image i text embeddinge
    # projektuje u zajednički embedding prostor.
    model = MultimodalModel()

    # Model se prebacuje na izabrani uređaj (CPU ili GPU).
    model.to(device)

    # Kreira se trainer koji upravlja procesom treniranja,
    # uključujući optimizator i contrastive loss.
    trainer = MultimodalTrainer(
        model=model,
        learning_rate=learning_rate,
        temperature=temperature,
    )

    # Model se trenira kroz zadati broj epoha.
    for epoch in range(1, epochs + 1):
        loss = trainer.train_epoch(
            dataloader,
            device=device,
        )

        # Ispis loss vrednosti omogućava praćenje napretka treniranja.
        print(
            f"Epoch {epoch}/{epochs} - loss: {loss:.4f}",
            flush=True,
        )

    # Kreira se direktorijum za izlazni model ukoliko već ne postoji.
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Čuvaju se naučeni parametri multimodalnog modela.
    torch.save(
        model.state_dict(),
        output_path,
    )

    print(
        f"Model saved to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    # Pokretanje treninga nad prethodno generisanim train embeddingima.
    train(
        embeddings_path="embeddings/train.pt",
        output_path="embeddings/multimodal_model.pt",
        epochs=10,
        batch_size=64,
        learning_rate=1e-3,
        temperature=0.07,
    )
