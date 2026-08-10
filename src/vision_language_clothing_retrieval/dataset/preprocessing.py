import random
import re

from vision_language_clothing_retrieval.dataset.sample import DatasetSample


def split_samples(
    samples: list[DatasetSample],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[
    list[DatasetSample],
    list[DatasetSample],
    list[DatasetSample],
]:
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than 1.")

    valid_samples = [sample for sample in samples if sample.text.strip()]

    groups: dict[str, list[DatasetSample]] = {}

    for sample in valid_samples:
        match = re.search(r"id_\d+", sample.sample_id)

        if match is None:
            raise ValueError(f"Could not extract item ID from: {sample.sample_id}")

        item_id = match.group()
        groups.setdefault(item_id, []).append(sample)

    item_ids = list(groups.keys())

    random.Random(seed).shuffle(item_ids)

    train_end = int(len(item_ids) * train_ratio)
    validation_end = int(len(item_ids) * (train_ratio + validation_ratio))

    train_ids = item_ids[:train_end]
    validation_ids = item_ids[train_end:validation_end]
    test_ids = item_ids[validation_end:]

    train = [
        sample
        for item_id in train_ids
        for sample in groups[item_id]
    ]

    validation = [
        sample
        for item_id in validation_ids
        for sample in groups[item_id]
    ]

    test = [
        sample
        for item_id in test_ids
        for sample in groups[item_id]
    ]

    return train, validation, test
