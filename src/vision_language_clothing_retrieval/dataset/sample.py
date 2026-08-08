from dataclasses import dataclass


@dataclass
class DatasetSample:
    sample_id: str
    image_path: str
    text: str
