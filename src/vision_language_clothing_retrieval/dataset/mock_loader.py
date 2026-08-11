from collections.abc import Iterable

from vision_language_clothing_retrieval.dataset.loader import DatasetLoader
from vision_language_clothing_retrieval.dataset.sample import DatasetSample


class MockDatasetLoader(DatasetLoader):
    """Testna implementacija loadera sa unapred definisanim uzorcima."""

    def load(self) -> Iterable[DatasetSample]:
        return [
            DatasetSample(
                sample_id="1",
                image_path="data/mock/red_shirt.jpg",
                text="red shirt",
            ),
            DatasetSample(
                sample_id="2",
                image_path="data/mock/blue_pants.jpg",
                text="blue pants",
            ),
        ]
