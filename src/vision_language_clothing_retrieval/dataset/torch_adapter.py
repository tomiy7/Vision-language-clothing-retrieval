from collections.abc import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from vision_language_clothing_retrieval.dataset.sample import DatasetSample


class TorchClothingDataset(Dataset):
    """Adapter koji omogućava korišćenje dataset uzoraka sa PyTorch DataLoader-om."""

    def __init__(self, samples: list[DatasetSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self._samples[index]


class MultimodalCollator:
    """Priprema slike i tekst za multimodalni PyTorch batch."""

    def __init__(self) -> None:
        # Sve slike se svode na istu dimenziju i pretvaraju u tensor
        # kako bi mogle da se obrađuju u batch-u.
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # Tokenizer priprema tekstualne opise za DistilBERT model.
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def __call__(
        self,
        samples: Sequence[DatasetSample],
    ) -> dict:
        image_tensors = []

        # Učitavanje i preprocesiranje svih slika iz trenutnog batch-a.
        for sample in samples:
            image = Image.open(sample.image_path).convert("RGB")
            image_tensors.append(self.image_transform(image))

        # Tekstualni opisi se obrađuju zajedno kako bi tokenizer
        # napravio odgovarajuće batch tenzore.
        texts = [sample.text for sample in samples]

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Batch zadržava identifikatore uzoraka, slike i tekstualne
        # reprezentacije potrebne za dalju multimodalnu obradu.
        return {
            "sample_ids": [sample.sample_id for sample in samples],
            "images": torch.stack(image_tensors),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
