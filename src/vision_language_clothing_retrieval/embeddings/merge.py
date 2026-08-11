from pathlib import Path

import torch


def merge_embeddings(
    input_dir: str,
    output_file: str,
) -> None:
    """Spaja embeddinge iz više checkpoint fajlova u jedan fajl."""

    input_path = Path(input_dir)
    output_path = Path(output_file)

    # Pronalaze se svi checkpoint fajlovi generisani tokom embedding procesa.
    checkpoints = sorted(input_path.glob("part_*.pt"))

    if not checkpoints:
        raise ValueError(f"No checkpoints found in {input_path}")

    sample_ids = []
    image_embeddings = []
    text_embeddings = []

    # Svaki checkpoint sadrži identifikatore, image embeddinge i text embeddinge.
    for checkpoint in checkpoints:
        data = torch.load(checkpoint, weights_only=True)

        sample_ids.extend(data["sample_ids"])
        image_embeddings.append(data["image_embeddings"])
        text_embeddings.append(data["text_embeddings"])

    # Embeddingi iz svih checkpoint-a spajaju se u jedinstvene tenzore.
    merged = {
        "sample_ids": sample_ids,
        "image_embeddings": torch.cat(image_embeddings),
        "text_embeddings": torch.cat(text_embeddings),
    }

    # Kreira se izlazni direktorijum ako već ne postoji.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)

    print(f"Merged {len(checkpoints)} checkpoints")
    print(f"Samples: {len(sample_ids)}")
    print(f"Image embeddings: {merged['image_embeddings'].shape}")
    print(f"Text embeddings: {merged['text_embeddings'].shape}")
    print(f"Saved to: {output_path}")
