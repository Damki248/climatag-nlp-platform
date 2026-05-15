# ClimaTag – Climate NLP Platform

An end-to-end web platform for **Named Entity Recognition (NER)** in the climate science domain, featuring human-in-the-loop annotation, model fine-tuning, and experiment tracking.

Built as part of a Master's thesis at the Faculty of Informatics and Digital Technologies, University of Rijeka.

> **Supervisor:** Prof. dr. sc. Sanda Martinčić-Ipšić

---

## What the platform does

- **NER** – Recognises 28 climate-domain entity categories (chemicals, locations, organisms, quantities, etc.) plus a custom **Climate Model** category (CMIP6, ERA5, SSP scenarios, etc.) using GLiNER
- **Human-in-the-loop annotation** – Model pre-annotates text; user corrects in the ClimaTag UI; corrections are pushed to Label Studio for storage
- **In-app fine-tuning** – Retrain the GLiNER model on new annotations directly from the UI, with live progress tracking
- **Experiment tracking** – All training runs logged to MLflow (metrics, hyperparameters, Climate Model F1)

---

## Architecture

```
┌─────────────────────────────────────────────┐
│           React Frontend (ClimaTag)          │
│   NER  │  Annotate  │  Train  │  Experiments │
└──────────────────────┬──────────────────────┘
                       │ HTTP (port 8000)
┌──────────────────────▼──────────────────────┐
│              FastAPI Backend                 │
│  /api/ner  │  /api/annotation  │  /api/train │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│           GLiNER NER Model                   │
│  Baseline (urchade/gliner_medium-v2.1)       │
│  Climate Model (fine-tuned on annotations)   │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│           Docker services                    │
│   Label Studio :8080  │  MLflow :5000        │
└─────────────────────────────────────────────┘
```

---

## Requirements

| Requirement | Version |
|---|---|
| OS | Linux or WSL2 (Ubuntu 22.04 recommended) |
| Python | 3.10 (via conda) |
| CUDA | 12.1 (GPU recommended, CPU works but slow) |
| Docker | 24+ with Docker Compose v2 |

> **Windows users:** Run everything inside WSL2.

---

## Project structure

```
climate-nlp-platform/
├── backend/
│   └── app/
│       ├── api/               # FastAPI routers (ner, annotation, train)
│       ├── services/          # Business logic (ner_service, label_studio_service)
│       └── main.py            # App entry point + static file serving
├── frontend/                  # React + Vite + Tailwind (ClimaTag UI)
│   └── dist/                  # Production build (served by FastAPI)
├── training/
│   └── ner_gliner/
│       └── train.py           # GLiNER fine-tuning script
├── docker/
│   └── docker-compose.yml     # Label Studio + MLflow containers
├── data/
│   ├── raw/CliReNER_SILVER/   # SILVER NER dataset (parquet)
│   └── annotations/           # Label Studio exports
├── models/                    # Model weights (not in Git – too large)
├── .env.example               # Environment variable template
└── README.md
```

> `models/` weights are excluded from Git. See [SETUP.md](SETUP.md) for download instructions.

---

## Quick start

See **[SETUP.md](SETUP.md)** for full setup instructions.

After setup, the platform runs on a single port:

```bash
# Start Docker services
docker compose -f docker/docker-compose.yml up -d

# Start backend (serves frontend + API)
conda activate climtag-env
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

| Service | URL |
|---|---|
| ClimaTag UI | http://localhost:8000 |
| FastAPI docs | http://localhost:8000/docs |
| Label Studio | http://localhost:8080 |
| MLflow | http://localhost:5000 |

---

## Environment variables

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `LABEL_STUDIO_TOKEN` | API token from Label Studio (Account → Access Token) |
| `LABEL_STUDIO_URL` | Default: `http://localhost:8080` |
| `LABEL_STUDIO_PROJECT_ID` | Project ID in Label Studio (usually `1`) |
| `MLFLOW_TRACKING_URI` | Default: `http://localhost:5000` |
| `ALLOWED_ORIGINS` | CORS origins, comma-separated. Default: `http://localhost:5173` |

---

## Training

See **[TRAINING.md](TRAINING.md)** for GLiNER fine-tuning instructions (CLI and in-app).
