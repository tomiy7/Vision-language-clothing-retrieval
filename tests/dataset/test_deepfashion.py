from vision_language_clothing_retrieval.dataset.deepfashion import (
    DeepFashionDatasetLoader,
)


IMAGE_DIR = "/home/vladimir/Downloads/images"
CAPTIONS_PATH = "/home/vladimir/Downloads/captions.json"


def test_deepfashion_loader():
    loader = DeepFashionDatasetLoader(
        image_dir=IMAGE_DIR,
        captions_path=CAPTIONS_PATH,
    )

    samples = loader.load()

    first = next(samples)

    assert first.sample_id
    assert first.image_path.endswith(".jpg")
    assert first.text
