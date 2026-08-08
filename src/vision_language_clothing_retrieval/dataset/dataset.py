from collections.abc import Sequence

from vision_language_clothing_retrieval.dataset.sample import DatasetSample


class ClothingDataset:
    def __init__(self, samples: Sequence[DatasetSample]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self._samples[index]
