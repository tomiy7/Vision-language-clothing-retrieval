from pathlib import Path

import torch
from torch.utils.data import DataLoader


class EmbeddingGenerator:
    """Generiše image i text embeddinge za uzorke iz DataLoader-a."""

    def __init__(
        self,
        image_model,
        text_model,
        device: str = "cpu",
    ) -> None:
        # Modeli se prebacuju na izabrani uređaj kako bi se inference
        # izvršavao na CPU-u ili GPU-u, u zavisnosti od podešavanja.
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)

        # Modeli se koriste samo za inference, bez ažuriranja parametara.
        self.image_model.eval()
        self.text_model.eval()

        self.device = device

    def generate(self, dataloader: DataLoader) -> dict:
        """Generiše embeddinge za ceo dataset u batch režimu."""
        sample_ids = []
        image_embeddings = []
        text_embeddings = []

        total_batches = len(dataloader)

        # Dataset se obrađuje batch po batch kako bi se kontrolisala
        # potrošnja memorije i omogućila efikasna obrada većeg broja uzoraka.
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

        # Embeddingi pojedinačnih batch-eva spajaju se u konačne tenzore.
        return {
            "sample_ids": sample_ids,
            "image_embeddings": torch.cat(image_embeddings),
            "text_embeddings": torch.cat(text_embeddings),
        }

    def generate_batch(
        self,
        batch: dict,
    ) -> dict:
        """Generiše image i text embeddinge za jedan batch."""

        images = batch["images"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Prilikom generisanja embeddinga nisu potrebni gradijenti,
        # čime se smanjuje memorijska potrošnja tokom inference-a.
        with torch.no_grad():
            image_embeddings = self.image_model(images)

            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Reprezentacija prvog tokena koristi se kao embedding
            # kompletnog tekstualnog opisa.
            text_embeddings = text_outputs.last_hidden_state[:, 0, :]

        # Embeddingi se vraćaju na CPU kako bi se oslobodila memorija
        # uređaja i omogućilo njihovo dalje čuvanje i obrada.
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
        """Generiše embeddinge i periodično ih čuva na disk."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        start_batch = 0

        if resume:
            # Ako postoje prethodno sačuvani checkpoint-i, pronalazi se
            # poslednji obrađeni batch kako bi se generisanje moglo nastaviti.
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

        # Nastavlja se obrada od prvog batch-a koji nije sačuvan
        # u prethodnom checkpoint-u.
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

            # Embeddingi se periodično čuvaju kako bi se izbegao gubitak
            # već obrađenih podataka ukoliko se proces prekine.
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

                # Nakon čuvanja checkpoint-a liste se prazne kako bi se
                # naredni deo embeddinga obrađivao nezavisno.
                batch_ids = []
                image_embeddings = []
                text_embeddings = []
