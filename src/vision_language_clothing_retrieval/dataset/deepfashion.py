import json
from pathlib import Path
from typing import Iterator

from vision_language_clothing_retrieval.dataset.loader import DatasetLoader
from vision_language_clothing_retrieval.dataset.sample import DatasetSample


class DeepFashionDatasetLoader(DatasetLoader):
    def __init__(
        self,
        image_dir: str,
        captions_path: str,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.captions_path = Path(captions_path)

    def load(self) -> Iterator[DatasetSample]:
        with self.captions_path.open("r", encoding="utf-8") as file:
            captions = json.load(file)

        for image_name, caption in captions.items():
            image_path = self.image_dir / image_name

            yield DatasetSample(
                sample_id=Path(image_name).stem,
                image_path=str(image_path),
                text=caption,
            )
