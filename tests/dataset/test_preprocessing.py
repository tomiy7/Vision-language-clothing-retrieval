import re

from vision_language_clothing_retrieval.dataset.deepfashion import (
    DeepFashionDatasetLoader,
)
from vision_language_clothing_retrieval.dataset.preprocessing import split_samples


IMAGE_DIR = "/home/vladimir/Downloads/images"
CAPTIONS_PATH = "/home/vladimir/Downloads/captions.json"


def get_item_ids(samples):
    return {
        re.search(r"id_\d+", sample.sample_id).group()
        for sample in samples
    }


def test_split_samples():
    loader = DeepFashionDatasetLoader(
        image_dir=IMAGE_DIR,
        captions_path=CAPTIONS_PATH,
    )

    samples = list(loader.load())

    train, validation, test = split_samples(
        samples,
        seed=42,
    )

    # Empty captions must not appear in the split.
    assert all(sample.text.strip() for sample in train)
    assert all(sample.text.strip() for sample in validation)
    assert all(sample.text.strip() for sample in test)

    # All valid samples must be included exactly once.
    assert len(train) + len(validation) + len(test) == 42537

    # The same clothing item must not appear in multiple splits.
    train_ids = get_item_ids(train)
    validation_ids = get_item_ids(validation)
    test_ids = get_item_ids(test)

    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids

def test_split_is_deterministic():
    loader = DeepFashionDatasetLoader(
        image_dir=IMAGE_DIR,
        captions_path=CAPTIONS_PATH,
    )

    samples = list(loader.load())

    first_split = split_samples(samples, seed=42)
    second_split = split_samples(samples, seed=42)

    assert first_split == second_split
