from dataclasses import dataclass


@dataclass
class DatasetSample:
    """Predstavlja jedan uzorak dataseta sa slikom i tekstualnim opisom."""

    sample_id: str
    image_path: str
    text: str
