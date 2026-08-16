# Vision-Language Clothing Retrieval

Multimodalni sistem za pretragu odeće koji povezuje vizuelne i tekstualne informacije. Projekat obuhvata učitavanje i preprocesiranje skupa podataka, podelu bez data leakage-a na train, validation i test skupove, generisanje image i text embeddinga i pripremu podataka za downstream retrieval komponentu.

## Autori

- Vladimir Mandić
- Milica Tošić
- Radenko Nikolić

---

## Opis projekta

Cilj projekta je razvoj sistema za cross-modal retrieval odeće, odnosno pronalaženje odgovarajućih slika na osnovu tekstualnog opisa i tekstualnih opisa na osnovu slike.

Projekat je organizovan kao pipeline koji obuhvata:

1. učitavanje skupa podataka,
2. preprocesiranje slika i tekstualnih opisa,
3. podelu podataka na train, validation i test skup,
4. generisanje image embeddinga,
5. generisanje text embeddinga,
6. čuvanje i organizaciju embeddinga,
7. pripremu reprezentacija za retrieval komponentu.

Za vizuelnu reprezentaciju koristi se ResNet10, dok se za tekstualnu reprezentaciju koristi pretrained DistilBERT model.

---

## Skup podataka

Projekat koristi skup slika odeće praćenih tekstualnim opisima.

Svaki uzorak sadrži:

- sliku,
- tekstualni opis,
- jedinstveni identifikator uzorka.

Nazivi slika sadrže identifikatore artikala, koji omogućavaju grupisanje različitih prikaza istog komada odeće.

Ukupan broj učitanih uzoraka je:

- **42.544 uzorka**
- **7 uzoraka sa praznim captionom**
- **42.537 validnih uzoraka**

Uzorci sa praznim captionom ne uključuju se u konačnu podelu na train, validation i test skupove.

### Podela skupa

| Skup       | Broj uzoraka |
| ---------- | -----------: |
| Train      |       34.047 |
| Validation |        4.354 |
| Test       |        4.136 |
| **Ukupno** |   **42.537** |

Podela se vrši na nivou artikla odeće kako bi se sprečio data leakage. Različiti prikazi istog artikla ne mogu završiti u različitim podskupovima.

Preklapanje između podskupova je provereno i iznosi:

```text
Train ∩ Validation = 0
Train ∩ Test       = 0
Validation ∩ Test  = 0
```

## Arhitektura

Pipeline projekta je organizovan tako da se obrada slike i tekstualnog opisa odvija kroz odvojene grane, nakon čega se dobijene reprezentacije prosleđuju narednoj retrieval komponenti.

                         Skup podataka
                              │
                              ▼
                     Preprocesiranje
                              │
                  Train / Validation / Test
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
               Slika                     Caption
                 │                         │
                 ▼                         ▼
             ResNet10                 DistilBERT
                 │                         │
                 ▼                         ▼
        Image embedding             Text embedding
           [512 dim.]                  [768 dim.]
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    Embedding storage
                              │
                              ▼
                  Retrieval komponenta

## Korišćeni modeli

### ResNet10

ResNet10 se koristi za generisanje vizuelnih reprezentacija slika.

Ulazna slika se:

1. konvertuje u RGB format,
2. menja na dimenziju `224 × 224`,
3. pretvara u PyTorch tensor,
4. prosleđuje ResNet10 modelu.

Za svaku sliku generiše se embedding dimenzije:

```text
512
```

### DistilBERT

Za tekstualne opise koristi se pretrained model:

```text
distilbert-base-uncased
```

Tekst se tokenizuje uz padding i truncation.

Kao tekstualna reprezentacija koristi se reprezentacija prvog tokena:

```python
text_outputs.last_hidden_state[:, 0, :]
```

Dobijeni text embedding ima dimenziju:

```text
768
```

## Preprocesiranje

### Slike

Slike se pre učitavanja u model konvertuju u RGB format i menjaju na dimenziju `224 × 224`.

Za transformaciju slike koriste se:

- `Resize((224, 224))`
- `ToTensor()`

Na ovaj način se svaka slika pretvara u PyTorch tensor oblika:

```text
[3, 224, 224]
```

### Tekst

Tekstualni opisi se obrađuju pomoću DistilBERT tokenizer-a.

Za svaki batch tekstova koristi se:

- padding,
- truncation,
- konverzija tokena u PyTorch tenzore.

Tokenizer kao rezultat generiše:

```text
input_ids
attention_mask
```

`input_ids` predstavljaju tokenizovane tekstualne opise, dok `attention_mask` označava stvarne tokene u odnosu na padding tokene.

### Batch obrada

Slike i tekst se obrađuju zajedno kroz `MultimodalCollator`, koji priprema podatke za PyTorch `DataLoader`.

Za svaki batch vraćaju se:

- `sample_ids`
- `images`
- `input_ids`
- `attention_mask`

Na ovaj način se za svaki uzorak zadržava veza između slike, tekstualnog opisa i njihovog identifikatora.

## Instalacija i pokretanje

Projekat koristi Poetry za upravljanje Python okruženjem i zavisnostima.

### Instalacija zavisnosti

```bash
poetry install
```
### Potrebni paketi

Glavne Python biblioteke koje projekat koristi su:

- `torch` – rad sa tenzorima i generisanje embeddinga
- `torchvision` – obrada slika i ResNet model
- `transformers` – DistilBERT model i tokenizer
- `Pillow` – učitavanje i obrada slika
- `numpy` – rad sa vektorskim reprezentacijama
- `pytest` – pokretanje testova
- `jupyter` – pokretanje Jupyter notebook-ova

Sve zavisnosti i njihove verzije definisane su u `pyproject.toml`.

### Aktiviranje okruženja

```bash
poetry shell
```

Alternativno, komande se mogu pokretati direktno preko Poetry-ja:

```bash
poetry run <komanda>
```

### Pokretanje Jupyter Notebook-a

```bash
poetry run jupyter notebook
```

Notebook-ovi se nalaze u direktorijumu:

```text
notebooks/
```

### Provera instaliranog okruženja

Verzija Python-a može se proveriti komandom:

```bash
poetry run python --version
```

A instalirane Jupyter komponente:

```bash
poetry run jupyter --version
```

### Pokretanje testova

Testovi se pokreću iz korena projekta komandom:

```bash
poetry run pytest
```

Na ovaj način se proverava ispravnost implementiranih komponenti nakon podešavanja okruženja.

## Struktura projekta

```text
.
├── data/
│   ├── images/
│   └── captions.json
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   └── 02_embedding_generation_demo.ipynb
├── src/
│   └── vision_language_clothing_retrieval/
│       ├── dataset/
│       └── embeddings/
├── tests/
│   ├── dataset/
│   └── embeddings/
├── pyproject.toml
├── poetry.lock
├── README.md
└── .gitignore
```

## Literatura

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
   https://arxiv.org/abs/1512.03385

2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv.
   https://arxiv.org/abs/1910.01108

3. Wolf, T., Debut, L., Sanh, V., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations.
   https://arxiv.org/abs/1910.03771

4. Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems (NeurIPS).
   https://arxiv.org/abs/1912.01703

5. Liu, Z., Luo, P., Qiu, S., Wang, X., & Tang, X. (2016). *DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
   [https://arxiv.org/abs/1605.07396](https://arxiv.org/abs/1605.07396)
