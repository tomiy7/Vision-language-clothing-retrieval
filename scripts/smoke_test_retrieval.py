from vision_language_clothing_retrieval.services.retrieval_service import (
    RetrievalService,
)

def main():
    service = RetrievalService(
        model_path="embeddings/multimodal_model.pt",
        embeddings_path="embeddings/test.pt",
        image_dir="data/images",
        captions_path="data/captions.json",
    )

    print("RetrievalService uspešno učitan.")
    print(f"Broj kandidata: {len(service.image_embeddings)}")

    query = "His shirt has long sleeves, cotton fabric and striped patterns. The neckline of it is round. The trousers this person wears is of long length. The trousers are with cotton fabric and solid color patterns. The outer clothing is with leather fabric and solid color patterns. There is an accessory on his wrist."

    results = service.retrieve_images(
        text=query,
        top_k=5,
    )

    print("\nQuery:")
    print(query)

    print("\nTop 5 rezultata:")
    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}.")
        print(f"sample_id: {result['sample_id']}")
        print(f"image_path: {result['image_path']}")
        print(f"text: {result['text']}")
        print(f"score: {result['score']:.4f}")


if __name__ == "__main__":
    main()