from PIL import Image

from vision_language_clothing_retrieval.embeddings.image_encoder import (
    ResNet10ImageEncoder,
)


def test_resnet10_image_encoder():
    image_path = "/tmp/test_image.jpg"

    Image.new("RGB", (500, 300), "red").save(image_path)

    encoder = ResNet10ImageEncoder()
    embedding = encoder.encode(image_path)

    assert isinstance(embedding, list)
    assert len(embedding) == 512
