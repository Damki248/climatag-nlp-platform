# Climate NLP Platform – Upute za pokretanje

> Magistarski rad – Damjan  
> Fakultet informatike i digitalnih tehnologija, Rijeka  
> Mentorica: Prof. dr. sc. Sanda Martinčić-Ipšić

---

## Sadržaj

1. [Preduvjeti](#preduvjeti)
2. [Conda environment](#conda-environment)
3. [Pokretanje Docker servisa](#pokretanje-docker-servisa)
4. [Pokretanje FastAPI backenda](#pokretanje-fastapi-backenda)
5. [Testiranje NER endpointa](#testiranje-ner-endpointa)
6. [Struktura projekta](#struktura-projekta)
7. [Poznati problemi](#poznati-problemi)

---

## Preduvjeti

- Windows 11 + WSL2 (Ubuntu 22.04)
- Docker Desktop s WSL2 integracijom
- NVIDIA GPU driver 591.86+ / CUDA 13.1
- Miniconda (conda 26.1.1+)
- VS Code s WSL ekstenzijom

---

## Conda environment

> ⚠️ Koristi `climtag-env` (Python 3.10) – **ne** `climate-nlp` (Python 3.11)!  
> `span_marker` biblioteka ne radi s Python 3.11.

```bash
# Aktivacija environmenta
conda activate climtag-env

# Provjera
python -c "import torch; print(torch.cuda.is_available())"  # mora biti True
```

### Ključne verzije paketa

| Paket | Verzija |
|-------|---------|
| Python | 3.10 |
| torch | 2.5.1+cu121 |
| transformers | 4.50.0 |
| tokenizers | 0.21.4 |
| span_marker | 1.7.0 |
| datasets | 3.0.0 |
| fastapi | 0.135.3 |
| mlflow | 3.11.1 |
| peft | 0.18.1 |
| accelerate | 1.13.0 |

---

## Pokretanje Docker servisa

```bash
cd ~/climate-nlp-platform

# Pokretanje (Label Studio + MLflow)
docker compose -f docker/docker-compose.yml up -d

# Provjera statusa
docker ps

# Zaustavljanje
docker compose -f docker/docker-compose.yml down
```

### URL-ovi

| Servis | URL |
|--------|-----|
| Label Studio | http://localhost:8080 |
| MLflow | http://localhost:5000 |

---

## Pokretanje FastAPI backenda

```bash
cd ~/climate-nlp-platform
conda activate climtag-env

uvicorn backend.app.main:app --reload --port 8000
```

Pri startu server automatski učitava NER model u memoriju (~30 sekundi).  
Pričekaj poruku: `NER model ready!`

### URL-ovi

| Servis | URL |
|--------|-----|
| Health check | http://localhost:8000/health |
| Swagger UI | http://localhost:8000/docs |
| NER endpoint | http://localhost:8000/api/ner/predict |

---

## Testiranje NER endpointa

### curl

```bash
curl -X POST http://localhost:8000/api/ner/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Climate change is causing rapid melting of Arctic ice sheets, leading to rising sea levels."}'
```

### Očekivani odgovor

```json
{
  "entities": [
    {"span": "Climate change", "label": "Meteorological Phenomenon", "score": 0.685, "start": 0, "end": 14},
    {"span": "Arctic ice sheets", "label": "Geographical Feature", "score": 0.476, "start": 43, "end": 60},
    {"span": "sea levels", "label": "Quantity", "score": 0.902, "start": 80, "end": 90}
  ],
  "count": 3
}
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/ner/predict",
    json={"text": "Climate change affects Arctic ecosystems."}
)
print(response.json())
```

---

## Struktura projekta

```
climate-nlp-platform/
├── backend/
│   └── app/
│       ├── main.py              # FastAPI app + lifespan (model loading)
│       ├── api/
│       │   └── ner.py           # NER router (/api/ner/predict)
│       ├── services/
│       │   └── ner_service.py   # NERService singleton
│       ├── models/              # Pydantic / DB modeli
│       └── schemas/             # Request/Response sheme
├── frontend/
│   └── src/                     # React (TODO)
├── training/
│   ├── ner/                     # NER fine-tuning skripte (TODO)
│   ├── classification/          # Classification fine-tuning (TODO)
│   └── peft/                    # LoRA / Adapter skripte (TODO)
├── models/
│   └── ner_baseline/            # CliReNER-cliscibert_scivocab_uncased
│       ├── model.safetensors    # ~430 MB
│       ├── config.json
│       ├── tokenizer.json
│       └── vocab.txt
├── data/
│   ├── raw/                     # Originalni dataseti (gitignored)
│   ├── processed/               # Obrađeni dataseti (gitignored)
│   └── annotations/             # Label Studio exporti
├── annotation/                  # Label Studio konfiguracije
├── docker/
│   └── docker-compose.yml       # Label Studio + MLflow
└── docs/                        # Dijagrami i bilješke
```

---

## Poznati problemi

### span_marker ne radi s Python 3.11
**Problem:** `ValueError: text input must be of type str...` bez obzira na format inputa.  
**Rješenje:** Koristi `climtag-env` (Python 3.10).

### Docker permission denied
**Problem:** `permission denied while trying to connect to the Docker daemon socket`  
**Rješenje:**
```bash
sudo usermod -aG docker $USER
# Zatim u PowerShellu:
wsl --shutdown
# Otvori novi terminal
```

### uvicorn: command not found
**Problem:** Pokušaj pokretanja bez aktiviranog environmenta.  
**Rješenje:** `conda activate climtag-env` prije pokretanja.

---

## TODO – sljedeći koraci

- [ ] SciDCC dataset (čeka se od mentorice)
- [ ] Classification fine-tuning pipeline
- [ ] Classification API endpoint (`/api/classify`)
- [ ] Label Studio integracija
- [ ] React frontend
- [ ] Full fine-tuning vs PEFT usporedba
- [ ] MLflow experiment tracking integracija

---

*Zadnje ažuriranje: 7. travnja 2026.*
