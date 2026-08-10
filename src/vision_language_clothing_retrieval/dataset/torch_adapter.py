import torch
from collections.abc import Sequence

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from vision_language_clothing_retrieval.dataset.sample import DatasetSample


class TorchClothingDataset(Dataset):
    def __init__(self, samples: list[DatasetSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self._samples[index]


class MultimodalCollator:
    def __init__(self) -> None:
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

    def __call__(
        self,
        samples: Sequence[DatasetSample],
    ) -> dict:
        image_tensors = []

        for sample in samples:
            image = Image.open(sample.image_path).convert("RGB")
            image_tensors.append(self.image_transform(image))

        texts = [sample.text for sample in samples]

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "sample_ids": [sample.sample_id for sample in samples],
            "images": torch.stack(image_tensors),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
