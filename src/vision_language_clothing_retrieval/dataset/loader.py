from collections.abc import Iterable

from vision_language_clothing_retrieval.dataset.sample import DatasetSample


class DatasetLoader:
    def load(self) -> Iterable[DatasetSample]:
        """Učitaj dataset i vrati njegove uzorke."""
        raise NotImplementedError
