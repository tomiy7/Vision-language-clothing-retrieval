import random

class RetrievalService:
    _FAKE_SAMPLE_IDS = [
        "WOMEN-Dresses-id_00001234-01_1_front",
        "WOMEN-Dresses-id_00005678-02_1_front",
        "MEN-Jackets_Vests-id_00001111-01_1_front",
        "WOMEN-Blouses_Shirts-id_00002222-01_1_front",
        "MEN-Sweatshirts_Hoodies-id_00003333-01_1_front",
        "WOMEN-Cardigans-id_00004444-01_1_front",
        "WOMEN-Tees_Tanks-id_00005555-01_1_front",
        "MEN-Suiting-id_00006666-01_1_front",
    ]

    _FAKE_CAPTIONS = [
        "a red dress with long sleeves and a fitted waist",
        "a blue denim jacket with front pockets",
        "a white cotton t-shirt with a round neckline",
        "a black cardigan with buttons down the front",
        "a striped blouse with short sleeves",
    ]

    def __init__(self, *args, **kwargs) -> None:
        pass

    def search_by_text(self, text: str, top_k: int = 5) -> list[dict]:
        return self._fake_results(top_k, result_type="image")

    def search_by_image(self, image_path: str, top_k: int = 5) -> list[dict]:
        return self._fake_results(top_k, result_type="text")

    def _fake_results(self, top_k: int, result_type: str) -> list[dict]:
        n = min(top_k, len(self._FAKE_SAMPLE_IDS))
        sample_ids = random.sample(self._FAKE_SAMPLE_IDS, n)

        results = []
        for i, sample_id in enumerate(sample_ids):
            score = round(0.9 - i * 0.07, 4)

            if result_type == "image":
                content = f"/images/{sample_id}.jpg"
            else:
                content = random.choice(self._FAKE_CAPTIONS)

            results.append(
                {
                    "id": sample_id,
                    "score": score,
                    "content": content
                }
            )

        return results