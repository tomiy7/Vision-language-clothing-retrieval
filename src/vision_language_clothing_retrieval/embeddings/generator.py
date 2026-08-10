from pathlib import Path

import torch
from torch.utils.data import DataLoader


class EmbeddingGenerator:
    def __init__(
        self,
        image_model,
        text_model,
        device: str = "cpu",
    ) -> None:
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)

        self.image_model.eval()
        self.text_model.eval()

        self.device = device

    def generate(self, dataloader: DataLoader) -> dict:
        sample_ids = []
        image_embeddings = []
        text_embeddings = []

        total_batches = len(dataloader)

        for batch_index, batch in enumerate(dataloader, start=1):
            result = self.generate_batch(batch)

            sample_ids.extend(result["sample_ids"])
            image_embeddings.append(result["image_embeddings"])
            text_embeddings.append(result["text_embeddings"])

            if batch_index % 10 == 0 or batch_index == total_batches:
                print(
                    f"Processed {batch_index}/{total_batches} batches",
                    flush=True,
                )

        return {
            "sample_ids": sample_ids,
            "image_embeddings": torch.cat(image_embeddings),
            "text_embeddings": torch.cat(text_embeddings),
        }

    def generate_batch(
        self,
        batch: dict,
    ) -> dict:
        images = batch["images"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            image_embeddings = self.image_model(images)

            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            text_embeddings = text_outputs.last_hidden_state[:, 0, :]

        return {
            "sample_ids": batch["sample_ids"],
            "image_embeddings": image_embeddings.cpu(),
            "text_embeddings": text_embeddings.cpu(),
        }

    def generate_to_disk(
            self,
            dataloader: DataLoader,
            output_dir: str,
            checkpoint_batches: int = 25,
            resume: bool = True,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        start_batch = 0

        if resume:
            checkpoints = sorted(output_path.glob("part_*.pt"))

            if checkpoints:
                last_checkpoint = checkpoints[-1]
                start_batch = int(
                    last_checkpoint.stem.split("_")[1]
                )

                print(
                    f"Resuming after batch {start_batch}",
                    flush=True,
                )

        batch_ids = []
        image_embeddings = []
        text_embeddings = []

        total_batches = len(dataloader)

        for batch_index, batch in enumerate(
                dataloader,
                start=1,
        ):
            if batch_index <= start_batch:
                continue

            result = self.generate_batch(batch)

            batch_ids.extend(result["sample_ids"])
            image_embeddings.append(result["image_embeddings"])
            text_embeddings.append(result["text_embeddings"])

            if (
                    batch_index % checkpoint_batches == 0
                    or batch_index == total_batches
            ):
                checkpoint = {
                    "sample_ids": batch_ids,
                    "image_embeddings": torch.cat(image_embeddings),
                    "text_embeddings": torch.cat(text_embeddings),
                }

                checkpoint_path = (
                        output_path / f"part_{batch_index:05d}.pt"
                )

                torch.save(checkpoint, checkpoint_path)

                print(
                    f"Saved {batch_index}/{total_batches} batches "
                    f"to {checkpoint_path} "
                    f"({len(batch_ids)} samples)",
                    flush=True,
                )

                batch_ids = []
                image_embeddings = []
                text_embeddings = []
