import json
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
    train_embeddings_path: str,
    validation_embeddings_path: str,
    output_path: str,
    history_path: str = "embeddings/training_history.json",
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    temperature: float = 0.07,
    device: str = "cpu",
) -> None:
    """Trenira multimodalni model nad postojećim embeddingima."""

    # Učitavaju se prethodno generisani image i text embeddingi.
    train_embeddings = load_embeddings(train_embeddings_path)
    validation_embeddings = load_embeddings(validation_embeddings_path)

    train_dataset = EmbeddingDataset(train_embeddings)
    validation_dataset = EmbeddingDataset(validation_embeddings)

    # Train dataloader meša redosled uzoraka između epoha.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Validation dataloader ne meša podatke jer se koristi samo za merenje.
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
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

    best_validation_loss = float("inf")
    history = []

    # Model se trenira kroz zadati broj epoha.
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(
            train_dataloader,
            device=device,
        )

        validation_loss = trainer.evaluate_epoch(
            validation_dataloader,
            device=device,
        )

        # Ispis loss vrednosti omogućava praćenje napretka treniranja.
        print(
            f"Epoch {epoch}/{epochs} - "
            f"train_loss: {train_loss:.4f} - "
            f"val_loss: {validation_loss:.4f}",
            flush=True,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation_loss,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            Path(output_path).parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                output_path,
            )

            print(
                f"  -> Nov najbolji model (val_loss: {validation_loss:.4f}), "
                f"sačuvan u {output_path}",
                flush=True,
            )

    # Kreira se direktorijum za izlazni model ukoliko već ne postoji.
    Path(history_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # # Čuvaju se naučeni parametri multimodalnog modela.
    # torch.save(
    #     model.state_dict(),
    #     output_path,
    # )

    print(
        f"\nTraining završen. Najbolji val_loss: {best_validation_loss:.4f}",
        flush=True,
    )
    print(f"Istorija treninga sačuvana u {history_path}", flush=True)


if __name__ == "__main__":
    # Pokretanje treninga nad prethodno generisanim train embeddingima.
    train(
        train_embeddings_path="embeddings/train.pt",
        validation_embeddings_path="embeddings/validation.pt",
        output_path="embeddings/multimodal_model.pt",
        history_path="embeddings/training_history.json",
        epochs=30,
        batch_size=16,
        learning_rate=1e-3,
        temperature=0.15,
    )
