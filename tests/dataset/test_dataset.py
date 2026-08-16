from vision_language_clothing_retrieval.dataset.dataset import ClothingDataset
from vision_language_clothing_retrieval.dataset.loader import DatasetLoader
from vision_language_clothing_retrieval.dataset.sample import DatasetSample


def test_clothing_dataset():
    samples = [
        DatasetSample("1", "shirt.jpg", "red shirt"),
        DatasetSample("2", "pants.jpg", "blue pants"),
    ]

    dataset = ClothingDataset(samples)

    assert len(dataset) == 2
    assert dataset[0] == samples[0]
    assert dataset[1] == samples[1]


class TestLoader(DatasetLoader):
    def load(self):
        return [
            DatasetSample(
                sample_id="1",
                image_path="shirt.jpg",
                text="red shirt",
            )
        ]


def test_dataset_loader():
    loader = TestLoader()

    samples = list(loader.load())

    assert len(samples) == 1
    assert samples[0].sample_id == "1"
    assert samples[0].image_path == "shirt.jpg"
    assert samples[0].text == "red shirt"
