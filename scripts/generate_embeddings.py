"""
Generiše embeddings/train.pt, embeddings/validation.pt i embeddings/test.pt
za CEO dataset (za razliku od notebook-a 02, koji ovo radi samo na 8
uzoraka radi demonstracije).

Pokretanje (iz korena projekta):
    poetry run python scripts/generate_embeddings.py

Očekuje da postoje:
    data/images/        - sve slike (flat folder)
    data/captions.json  - {"ime_slike.jpg": "tekstualni opis", ...}

Napomena: ovo je sporo na CPU-u (42.5k uzoraka kroz ResNet10 + DistilBERT).
Za brzu proveru da pipeline radi, prvo pokreni sa smanjenim MAX_SAMPLES.
"""

import shutil
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import logging as hf_logging

from vision_language_clothing_retrieval.dataset.deepfashion import (
    DeepFashionDatasetLoader,
)
from vision_language_clothing_retrieval.dataset.preprocessing import (
    split_samples,
)
from vision_language_clothing_retrieval.dataset.torch_adapter import (
    MultimodalCollator,
    TorchClothingDataset,
)
from vision_language_clothing_retrieval.embeddings.generator import (
    EmbeddingGenerator,
)
from vision_language_clothing_retrieval.embeddings.image_encoder import (
    ResNet10ImageEncoder,
)
from vision_language_clothing_retrieval.embeddings.merge import (
    merge_embeddings,
)
from vision_language_clothing_retrieval.embeddings.text_encoder import (
    DistilBERTTextEncoder,
)

hf_logging.set_verbosity_error()

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
CAPTIONS_PATH = DATA_DIR / "captions.json"

EMBEDDINGS_DIR = Path("embeddings")
CHECKPOINTS_DIR = EMBEDDINGS_DIR / "_checkpoints"

BATCH_SIZE = 32
CHECKPOINT_BATCHES = 25
DEVICE = "cpu"  # promeni u "cuda" ako imaš GPU

# Postavi na broj (npr. 200) za brzu probu na malom podskupu, ili None
# za ceo dataset.
MAX_SAMPLES: int | None = 200#None

def generate_split(
    split_name: str,
    samples: list,
    image_encoder: ResNet10ImageEncoder,
    text_encoder: DistilBERTTextEncoder,
) -> None:
    print(f"\n=== {split_name.upper()} ({len(samples)} uzoraka) ===")

    dataset = TorchClothingDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=MultimodalCollator(),
    )

    generator = EmbeddingGenerator(
        image_model=image_encoder.model,
        text_model=text_encoder.model,
        device=DEVICE,
    )

    checkpoint_dir = CHECKPOINTS_DIR / split_name

    generator.generate_to_disk(
        dataloader,
        output_dir=str(checkpoint_dir),
        checkpoint_batches=CHECKPOINT_BATCHES,
        resume=True,
    )

    output_file = EMBEDDINGS_DIR / f"{split_name}.pt"
    merge_embeddings(
        input_dir=str(checkpoint_dir),
        output_file=str(output_file),
    )

    # Checkpoint fajlovi se brišu nakon uspešnog spajanja - nisu
    # potrebni u finalnom embeddings/ folderu.
    # shutil.rmtree(checkpoint_dir, ignore_errors=True)


def main() -> None:
    print("Učitavanje uzoraka...")
    loader = DeepFashionDatasetLoader(
        str(IMAGES_DIR),
        str(CAPTIONS_PATH),
    )
    samples = list(loader.load())
    print(f"Ukupno učitano: {len(samples)} uzoraka")

    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]
        print(f"MAX_SAMPLES podešeno - koristi se samo {len(samples)} uzoraka")

    train_samples, validation_samples, test_samples = split_samples(samples)

    print(f"Train: {len(train_samples)}")
    print(f"Validation: {len(validation_samples)}")
    print(f"Test: {len(test_samples)}")

    print("\nUčitavanje ResNet10 i DistilBERT enkodera...")
    image_encoder = ResNet10ImageEncoder()
    text_encoder = DistilBERTTextEncoder()

    for split_name, split_samples_list in (
        ("train", train_samples),
        ("validation", validation_samples),
        ("test", test_samples),
    ):
        generate_split(
            split_name,
            split_samples_list,
            image_encoder,
            text_encoder,
        )

    print("\nGotovo. Fajlovi sačuvani u embeddings/train.pt, "
          "embeddings/validation.pt, embeddings/test.pt")


if __name__ == "__main__":
    main()
