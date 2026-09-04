import torch

from vision_language_clothing_retrieval.embeddings.text_encoder import (
    DistilBERTTextEncoder,
)
from vision_language_clothing_retrieval.retrieval.multimodal_retriever import (
    MultimodalRetriever,
)

EMBEDDINGS_PATH = "embeddings/test.pt"
MODEL_PATH = "embeddings/multimodal_model.pt"


def main() -> None:
    embeddings = torch.load(EMBEDDINGS_PATH, weights_only=True)

    sample_ids = embeddings["sample_ids"]
    image_embeddings = embeddings["image_embeddings"]
    text_embeddings = embeddings["text_embeddings"]

    print(f"Test set: {len(sample_ids)} uzoraka")

    retriever = MultimodalRetriever(model_path=MODEL_PATH)
    text_encoder = DistilBERTTextEncoder()

    # text -> image
    query_text = "red dress with long sleeves"
    query_embedding = torch.tensor(text_encoder.encode(query_text)).unsqueeze(0)

    indices, scores = retriever.retrieve_image(
        query_embedding, image_embeddings, top_k=5
    )

    print(f"\nText -> Image | query: '{query_text}'")
    for idx, score in zip(indices.tolist(), scores.tolist()):
        print(f"  {score:.4f}  {sample_ids[idx]}")

    # image -> text, uzimamo prvu sliku iz test skupa kao query
    query_index = 0
    image_embedding = image_embeddings[query_index].unsqueeze(0)

    indices, scores = retriever.retrieve_text(
        image_embedding, text_embeddings, top_k=5
    )

    print(f"\nImage -> Text | query slika: {sample_ids[query_index]}")
    for idx, score in zip(indices.tolist(), scores.tolist()):
        hit = " <- tačan par" if idx == query_index else ""
        print(f"  {score:.4f}  {sample_ids[idx]}{hit}")


if __name__ == "__main__":
    main()
