import torch

from vision_language_clothing_retrieval.evaluation.text_to_image.evaluator import (
    TextToImageEvaluator,
)
from vision_language_clothing_retrieval.evaluation.image_to_text.evaluator import (
    ImageToTextEvaluator,
)
from vision_language_clothing_retrieval.retrieval.multimodal_retriever import (
    MultimodalRetriever,
)

if __name__ == "__main__":
    embeddings = torch.load("embeddings/test.pt", weights_only=True)

    retriever = MultimodalRetriever(model_path="embeddings/multimodal_model.pt")

    text_to_image_evaluator = TextToImageEvaluator(
        retriever=retriever,
        image_embeddings=embeddings["image_embeddings"],
        text_embeddings=embeddings["text_embeddings"],
    )

    image_to_text_evaluator = ImageToTextEvaluator(
        retriever=retriever,
        image_embeddings=embeddings["image_embeddings"],
        text_embeddings=embeddings["text_embeddings"],
    )

    t2i_results = text_to_image_evaluator.evaluate()
    i2t_results = image_to_text_evaluator.evaluate(k_values=(1, 5, 10))

    print("=== Text-to-Image Retrieval Evaluation (test set) ===")
    for name, value in t2i_results.items():
        print(f"{name}: {value:.4f}")

    print("\n=== Image-to-Text Retrieval Evaluation (test set) ===")
    for name, value in i2t_results.items():
        print(f"{name}: {value:.4f}")