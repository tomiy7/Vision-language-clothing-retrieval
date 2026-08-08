from PIL import Image

from vision_language_clothing_retrieval.embeddings.image_encoder import (
    ResNet10ImageEncoder,
)
from vision_language_clothing_retrieval.embeddings.multimodal_encoder import (
    MultimodalEncoder,
)
from vision_language_clothing_retrieval.embeddings.text_encoder import (
    DistilBERTTextEncoder,
)


def test_multimodal_encoder():
    image_path = "/tmp/test_image.jpg"
    Image.new("RGB", (500, 300), "red").save(image_path)

    encoder = MultimodalEncoder(
        ResNet10ImageEncoder(),
        DistilBERTTextEncoder(),
    )

    embedding = encoder.encode(image_path, "red shirt")

    assert isinstance(embedding, list)
    assert len(embedding) == 1280
