# Vision-Language Clothing Retrieval

Multimodalni sistem za pretragu odeće koji povezuje vizuelne i tekstualne informacije. Projekat obuhvata učitavanje i preprocesiranje skupa podataka, podelu bez data leakage-a na train, validation i test skupove, generisanje image i text embeddinga, projektovanje reprezentacija u zajednički embedding prostor, treniranje pomoću kontrastivnog učenja i evaluaciju cross-modal retrieval sistema.

## Autori

- Vladimir Mandić
- Milica Tošić
- Radenko Nikolić

---

## Opis projekta

Cilj projekta je razvoj sistema za cross-modal retrieval odeće, odnosno:

- pronalaženje odgovarajućih slika na osnovu tekstualnog opisa (**Text-to-Image Retrieval**),
- pronalaženje odgovarajućih tekstualnih opisa na osnovu slike (**Image-to-Text Retrieval**).

Projekat je organizovan kao pipeline koji obuhvata:

1. učitavanje skupa podataka,
2. preprocesiranje slika i tekstualnih opisa,
3. podelu podataka na train, validation i test skup,
4. generisanje image embeddinga,
5. generisanje text embeddinga,
6. čuvanje i organizaciju embeddinga,
7. projektovanje image i text embeddinga u zajednički prostor,
8. treniranje projekcionih slojeva pomoću contrastive loss funkcije,
9. Text-to-Image i Image-to-Text retrieval,
10. evaluaciju pomoću Recall@K, MRR i Mean Rank metrika.

Za vizuelnu reprezentaciju koristi se **ResNet10** inicijalizovan kompatibilnim pretrained težinama ResNet18 modela, dok se za tekstualnu reprezentaciju koristi pretrained **DistilBERT** model.

---

## Skup podataka

Projekat koristi skup slika odeće praćenih tekstualnim opisima.
[Link ka drajvu za preuzimanje podataka](https://drive.google.com/drive/folders/1An2c_ZCkeGmhJg0zUjtZF46vyJgQwIr2)  
**Napomena:** dovoljno je da se skinu captions.json i images.zip


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

| Skup | Broj uzoraka |
| --- | ---: |
| Train | 34.047 |
| Validation | 4.354 |
| Test | 4.136 |
| **Ukupno** | **42.537** |

Podela se vrši na nivou artikla odeće kako bi se sprečio data leakage. Različiti prikazi istog artikla ne mogu završiti u različitim podskupovima.

Preklapanje između podskupova je provereno i iznosi:

```text
Train ∩ Validation = 0
Train ∩ Test       = 0
Validation ∩ Test  = 0
```

---

## Arhitektura

Pipeline projekta sastoji se od dve grane: vizuelne i tekstualne.

Slike se obrađuju pomoću ResNet10 modela, dok se tekstualni opisi obrađuju pomoću DistilBERT modela. Dobijeni embedding vektori različitih dimenzija projektuju se u zajednički 256-dimenzionalni embedding prostor.

```text
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
                 ▼                         ▼
        Image projection            Text projection
                 │                         │
                 └────────────┬────────────┘
                              ▼
                  Zajednički embedding
                         [256 dim.]
                              │
                              ▼
                     Contrastive loss
                              │
                              ▼
                   Multimodal retrieval
                     ↙               ↘
             Text → Image       Image → Text
```

Cilj treniranja je da odgovarajući image-text parovi budu međusobno bliski u zajedničkom embedding prostoru, dok se reprezentacije nepovezanih uzoraka udaljavaju.

---

## Korišćeni modeli

### ResNet10

ResNet10 se koristi za generisanje vizuelnih reprezentacija slika.

Model koristi ResNet arhitekturu sa `BasicBlock` blokovima:

```python
[1, 1, 1, 1]
```

što odgovara ResNet10 arhitekturi.

Umesto potpuno nasumične inicijalizacije, kompatibilni slojevi ResNet10 modela inicijalizuju se pretrained težinama **ResNet18** modela treniranog na ImageNet skupu.

Pretrained ResNet18 koristi se kao izvor težina:

```python
pretrained_model = resnet18(
    weights=ResNet18_Weights.DEFAULT,
)
```

Kompatibilne težine se zatim učitavaju u ResNet10:

```python
model.load_state_dict(
    pretrained_model.state_dict(),
    strict=False,
)
```

Opcija `strict=False` omogućava učitavanje kompatibilnih parametara iako ResNet10 i ResNet18 nemaju potpuno identičnu arhitekturu.

Na ovaj način koristi se **transfer learning**: ResNet10 zadržava svoju arhitekturu, ali počinje od već naučenih vizuelnih reprezentacija umesto od potpuno nasumičnih težina.

Završni klasifikacioni sloj uklanja se:

```python
model.fc = torch.nn.Identity()
```

tako da model za svaku sliku generiše embedding dimenzije:

```text
512
```

### DistilBERT

Za tekstualne opise koristi se pretrained model:

```text
distilbert-base-uncased
```

Tekst se tokenizuje pomoću odgovarajućeg tokenizer-a uz:

- padding,
- truncation,
- attention mask.

Kao tekstualna reprezentacija koristi se reprezentacija prvog tokena:

```python
text_outputs.last_hidden_state[:, 0, :]
```

Dobijeni text embedding ima dimenziju:

```text
768
```

---

## Preprocesiranje

### Slike

Sve slike se prvo učitavaju i konvertuju u RGB format.

Za preprocessing se koriste transformacije povezane sa pretrained ResNet18 težinama:

```python
weights = ResNet18_Weights.DEFAULT
image_transform = weights.transforms()
```

Na ovaj način preprocessing odgovara preprocessing-u korišćenom za pretrained ImageNet model.

Transformacije obuhvataju odgovarajuću promenu veličine slike, konverziju u PyTorch tensor i normalizaciju pomoću ImageNet statistika.

Dobijene slike imaju format pogodan za prosleđivanje ResNet modelu.

### Tekst

Tekstualni opisi se obrađuju pomoću DistilBERT tokenizer-a:

```text
distilbert-base-uncased
```

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

Slike i tekst obrađuju se zajedno kroz `MultimodalCollator`, koji priprema podatke za PyTorch `DataLoader`.

Za svaki batch vraćaju se:

- `sample_ids`,
- `images`,
- `input_ids`,
- `attention_mask`.

Na ovaj način se tokom celog pipeline-a zadržava veza između slike, tekstualnog opisa i identifikatora uzorka.

---

## Generisanje embeddinga

Image i text embeddingi generišu se za train, validation i test skup.

Pokretanje generisanja embeddinga:

```bash
poetry run python scripts/generate_embeddings.py
```

Generisani embeddingi čuvaju se u:

```text
embeddings/train.pt
embeddings/validation.pt
embeddings/test.pt
```

Svaki fajl sadrži:

```text
sample_ids
image_embeddings
text_embeddings
```

Dimenzije reprezentacija su:

```text
Image embedding: 512
Text embedding:  768
```

Generisanje podržava periodično čuvanje checkpoint-a kako bi se obrada većeg skupa mogla nastaviti nakon eventualnog prekida.

**Napomena:** Generisaniembedding fajlovi dostupni za preuzimanje na sledećem linku: [Link ka embedding fajlovima](https://drive.google.com/drive/folders/1QoevwOn8KCeRXKG72I04Pl1SRyXLd-p_?usp=sharing)

---

## Multimodalna projekcija

Pošto image i text embeddingi imaju različite dimenzije, koriste se zasebni projekcioni slojevi koji ih preslikavaju u zajednički embedding prostor.

Ulazne dimenzije su:

```text
Image: 512
Text:  768
```

dok zajednički embedding prostor ima dimenziju:

```text
256
```

Za oba modaliteta koristi se projekciona mreža koja sadrži linearne slojeve, GELU aktivacionu funkciju i dropout.

Nakon projekcije, image i text reprezentacije imaju istu dimenziju i mogu se direktno porediti.

---

## Contrastive learning

Multimodalni model trenira se pomoću simetričnog contrastive loss-a.

Pre računanja sličnosti, image i text embeddingi se L2 normalizuju:

```python
F.normalize(embeddings, p=2, dim=-1)
```

Nakon toga se računa matrica sličnosti:

```python
logits = image_embeddings @ text_embeddings.T
```

Pošto su embeddingi normalizovani, skalarni proizvod odgovara cosine similarity vrednosti.

Similarity vrednosti dele se temperature parametrom:

```python
logits = logits / temperature
```

Loss se računa u oba smera:

- Image-to-Text,
- Text-to-Image.

Konačni loss predstavlja prosek ova dva pravca.

Na ovaj način model istovremeno uči da za sliku pronađe odgovarajući tekst i da za tekst pronađe odgovarajuću sliku.

---

## Treniranje

Multimodalni projekcioni model trenira se nad prethodno generisanim train embeddingima.

Pokretanje treninga:

```bash
poetry run python -m vision_language_clothing_retrieval.retrieval.train
```

Tokom treninga prati se loss na train i validation skupu.

Model sa najboljim validation loss-om čuva se u:

```text
embeddings/multimodal_model.pt
```

Implementiran je i **early stopping**, kojim se trening prekida ukoliko se validation loss ne poboljšava tokom unapred definisanog broja epoha.

Istorija treninga čuva se u:

```text
embeddings/training_history.json
```

## Rezultati najboljeg treninga

| Batch size | Learning rate | Temperature | Best validation loss | Best epoch | Model path |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 0.001 | 0.15 | **1.720189** | 27 | `embeddings/multimodal_model.pt` |

---

## Eksperimenti sa hiperparametrima

U `03_model_training_experiments.ipynb` izvršen je **Grid Search** nad različitim kombinacijama hiperparametara:
- `batch_size`
- `learning_rate`
- `temperature`

Za svaku konfiguraciju model je treniran i evaluiran na **validation skupu**. Kandidati su poređeni pomoću retrieval metrika, pre svega **MRR** i **Recall@10**, kako bi se izabrala konfiguracija koja daje najbolje performanse u samom retrieval zadatku.

`test.pt` skup nije korišćen tokom Grid Search-a niti pri izboru modela, već je ostavljen za konačnu evaluaciju modela u notebooku 04.

**Izabrani model za finalnu evaluaciju**

| Parametar | Vrednost |
| --- | --- |
| Model | **Candidate 1** |
| Batch size | **16** |
| Learning rate | **0.001** |
| Temperature | **0.15** |
| Najmanja greška na validaciji | **1.7202** |
| Najbolja epoha po validation loss-u | **27** |
---  


## Retrieval

Nakon treniranja, naučene projekcije koriste se za cross-modal retrieval.

Podržana su dva smera.

### Text-to-Image Retrieval

Za zadati tekstualni opis pronalaze se slike sa najsličnijim reprezentacijama u zajedničkom embedding prostoru.

```text
Text query
    ↓
Text embedding
    ↓
Text projection
    ↓
Cosine similarity
    ↓
Top-K images
```

### Image-to-Text Retrieval

Za zadatu sliku pronalaze se najsličniji tekstualni opisi.

```text
Image query
    ↓
Image embedding
    ↓
Image projection
    ↓
Cosine similarity
    ↓
Top-K captions
```

---

## Evaluacija

Evaluacija se vrši na izdvojenom test skupu koji nije korišćen za treniranje modela.

Pokretanje evaluacije:

```bash
poetry run python scripts/run_evaluation.py
```

Koriste se sledeće metrike:

- **Recall@1**
- **Recall@5**
- **Recall@10**
- **MRR (Mean Reciprocal Rank)**
- **Mean Rank**

### Recall@K

Recall@K meri koliko često se tačan rezultat nalazi među prvih `K` rezultata retrieval sistema.

Na primer, Recall@10 predstavlja udeo upita za koje se odgovarajući rezultat nalazi među prvih 10 pronađenih rezultata.

Veća vrednost predstavlja bolji rezultat.

### MRR

Mean Reciprocal Rank meri koliko visoko se u proseku nalazi prvi tačan rezultat.

Veća vrednost predstavlja bolji rezultat.

### Mean Rank

Mean Rank predstavlja prosečnu poziciju odgovarajućeg rezultata u rangiranoj listi.

Kod ove metrike je **manja vrednost bolja**.

---

## Rezultati finalne evaluacije

Konačni model, izabran tokom Grid Search eksperimenata, evaluiran je na izdvojenom test skupu od **4.136 uzoraka**.

Dimenzije embeddinga korišćenih tokom evaluacije:

| Reprezentacija | Dimenzija |
| --- | ---: |
| Image embedding | (4136, 512) |
| Text embedding | (4136, 768) |
| Projektovani image embedding | (4136, 256) |
| Projektovani text embedding | (4136, 256) |

Nakon projekcije, image i text embeddingi nalaze se u zajedničkom 256-dimenzionalnom embedding prostoru.

Između svih test image i text reprezentacija formirana je matrica cosine similarity vrednosti dimenzije (4136, 4136), sa vrednostima u rasponu od **-0.9108** do **0.9425**.

### Text-to-Image Retrieval

| Metrika | Rezultat |
| --- | ---: |
| Recall@1 | **0.0791** |
| Recall@5 | **0.2297** |
| Recall@10 | **0.3441** |
| MRR | **0.1629** |
| Mean Rank | **83.4894** |

### Image-to-Text Retrieval

| Metrika | Rezultat |
| --- | ---: |
| Recall@1 | **0.0624** |
| Recall@5 | **0.1973** |
| Recall@10 | **0.2926** |
| MRR | **0.1407** |
| Mean Rank | **97.6308** |

Dobijeni rezultati pokazuju da model uspešno uči zajednički embedding prostor i omogućava retrieval u oba smera: Text-to-Image i Image-to-Text.

Najbolji rezultati ostvareni su u smeru Text-to-Image, gde je Recall@10 **0.3441**, dok je za Image-to-Text Recall@10 **0.2926**.

**Posebno značajno poboljšanje dobijeno je korišćenjem transfer learning pristupa za vizuelni enkoder, odnosno inicijalizacijom kompatibilnih slojeva ResNet10 modela pretrained ResNet18 težinama i korišćenjem odgovarajućeg ImageNet preprocessing-a.**

Napomena: AUC/ROC-AUC evaluacija biće dodatno uključena nakon završetka odgovarajuće analize u notebooku 04.

---

## Instalacija i pokretanje

Projekat koristi Poetry za upravljanje Python okruženjem i zavisnostima.

### Instalacija zavisnosti

```bash
poetry install
```

### Potrebni paketi

Glavne Python biblioteke koje projekat koristi su:

- `torch` – rad sa tenzorima, modelima i treniranjem,
- `torchvision` – obrada slika i ResNet modeli,
- `transformers` – DistilBERT model i tokenizer,
- `Pillow` – učitavanje i obrada slika,
- `numpy` – rad sa numeričkim podacima,
- `pytest` – pokretanje testova,
- `jupyter` – pokretanje Jupyter notebook-ova.

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

---

## Pokretanje kompletnog pipeline-a

Nakon instalacije zavisnosti, kompletan pipeline može se izvršiti sledećim redosledom.

### 1. Generisanje embeddinga

```bash
poetry run python scripts/generate_embeddings.py
```

### 2. Treniranje multimodalnog modela

```bash
poetry run python -m vision_language_clothing_retrieval.retrieval.train
```

### 3. Evaluacija

```bash
poetry run python scripts/run_evaluation.py
```

Generisanje embeddinga može biti vremenski zahtevno, posebno kada se izvršava na CPU-u.

---

## Struktura projekta

```text
.
├── data/
│   ├── images/
│   └── captions.json
│
├── embeddings/
│   ├── train.pt
│   ├── validation.pt
│   ├── test.pt
│   ├── multimodal_model.pt
│   └── experiments/
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_embedding_generation_demo.ipynb
│   ├── 03_model_training_experiments.ipynb
│   └── 04_retrieval_evaluation.ipynb
│
├── scripts/
│   ├── generate_embeddings.py
│   ├── run_evaluation.py
│   ├── smoke_test_retrieval.py
│   └── test_retrieval_flow.py
│
├── src/
│   └── vision_language_clothing_retrieval/
│       ├── api/
│       │   ├── main.py
│       │   ├── routes.py
│       │   └── schemas.py
│       │
│       ├── dataset/
│       │   ├── dataset.py
│       │   ├── deepfashion.py
│       │   ├── loader.py
│       │   ├── mock_loader.py
│       │   ├── preprocessing.py
│       │   ├── sample.py
│       │   └── torch_adapter.py
│       │
│       ├── embeddings/
│       │   ├── generator.py
│       │   ├── image_encoder.py
│       │   ├── merge.py
│       │   ├── multimodal_encoder.py
│       │   ├── resnet.py
│       │   ├── storage.py
│       │   └── text_encoder.py
│       │
│       ├── evaluation/
│       │   ├── image_to_text/
│       │   ├── text_to_image/
│       │   └── metrics.py
│       │
│       ├── retrieval/
│       │   ├── embedding_dataset.py
│       │   ├── loss.py
│       │   ├── matching.py
│       │   ├── model.py
│       │   ├── multimodal_retriever.py
│       │   ├── projection.py
│       │   ├── retriever.py
│       │   ├── similarity.py
│       │   ├── train.py
│       │   └── trainer.py
│       │
│       └── services/
│           └── retrieval_service.py
│
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── styles.css
│
├── tests/
│
├── pyproject.toml
├── poetry.lock
├── README.md
└── .gitignore
```



Direktorijum `embeddings/` sadrži generisane podatke i istrenirane parametre modela i ne mora biti verzionisan kroz Git.

---

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
   https://arxiv.org/abs/1605.07396
