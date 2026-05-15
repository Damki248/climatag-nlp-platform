# ClimaTag – Climate NLP Platform

An end-to-end web platform for **Named Entity Recognition (NER)** and **Text Classification** in the climate science domain, with human-in-the-loop annotation and model retraining support.

Built as part of a Master's thesis at the Faculty of Informatics and Digital Technologies, University of Rijeka.

> **Supervisor:** Prof. dr. sc. Sanda Martinčić-Ipšić  
> **Models:** [BERTmosphere collection](https://huggingface.co/collections/P0L3/bertmosphere-681db99388ca86d430f14347) (CliReBERT, CliSciBERT, SciClimateBERT)

---

## What the platform does

- **NER** – Recognises 28 climate-domain entity categories (chemicals, locations, organisms, quantities, etc.) using the CliReNER dataset and a SpanMarker + CliSciBERT model
- **Text classification** – Classifies climate-related articles into 20 topic categories (SciDCC dataset) using SciClimateBERT
- **Human-in-the-loop annotation** – Model pre-annotates text; user corrects in the ClimaTag UI; corrections are pushed to Label Studio
- **Experiment tracking** – All training runs logged to MLflow (metrics, hyperparameters, per-class F1)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend (ClimaTag)          │
│         NER  │  Classify  │  Annotate               │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (port 8000)
┌──────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                      │
│   /api/ner/predict  │  /api/classify  │  /api/annotate│
└──────┬───────────────────────┬────────────────────────┘
       │                       │
┌──────▼──────┐       ┌────────▼────────┐
│  NER Model  │       │   CLS Model     │
│ (SpanMarker │       │ (SciClimateBERT │
│ +CliSciBERT)│       │  fine-tuned)    │
└─────────────┘       └─────────────────┘
       │                       │
┌──────▼───────────────────────▼─────────┐
│  Docker services                       │
│  Label Studio :8080 │ MLflow :5000     │
└────────────────────────────────────────┘
```

---

## Requirements

| Requirement | Version |
|---|---|
| OS | Linux or WSL2 (Ubuntu 22.04 recommended) |
| Python | 3.10 (via conda) |
| CUDA | 12.1 (GPU recommended, CPU works but slow) |
| Node.js | 18+ |
| Docker | 24+ with Docker Compose v2 |

> **Note for Windows users:** Run everything inside WSL2. Native Windows is not supported due to ML stack compatibility issues.

---

## Project structure

```
climate-nlp-platform/
├── backend/
│   └── app/
│       ├── api/           # FastAPI route handlers (ner, cls, annotation)
│       ├── services/      # Business logic (ner_service, cls_service, label_studio_service)
│       └── main.py        # App entry point
├── frontend/              # React + Vite + Tailwind app (ClimaTag UI)
├── training/
│   └── classification/
│       └── train.py       # SciDCC fine-tuning script (full FT + LoRA)
├── notebooks/             # EDA and preprocessing notebooks
├── docker/
│   └── docker-compose.yml # Label Studio + MLflow containers
├── data/                  # Datasets (not in Git – see below)
├── models/                # Trained model weights (not in Git – see below)
├── .env.example           # Environment variable template
└── README.md
```

> `data/` and `models/` are excluded from Git (too large). See [Setup](#setup) for how to obtain them.

---

## Setup

See **[SETUP.md](SETUP.md)** for full step-by-step instructions covering:
- Conda environment creation
- Model download
- Docker services
- Backend and frontend startup

---

## Quick start (after full setup)

```bash
# 1. Start Docker services (Label Studio + MLflow)
cd ~/climate-nlp-platform
docker compose -f docker/docker-compose.yml up -d

# 2. Start backend (new terminal)
conda activate climtag-env
cd ~/climate-nlp-platform
uvicorn backend.app.main:app --reload --port 8000

# 3. Start frontend (new terminal)
cd ~/climate-nlp-platform/frontend
npm run dev
```

| Service | URL |
|---|---|
| ClimaTag UI | http://localhost:5173 |
| FastAPI docs | http://localhost:8000/docs |
| Label Studio | http://localhost:8080 |
| MLflow | http://localhost:5000 |

---

## Training

See **[TRAINING.md](TRAINING.md)** for instructions on running classification fine-tuning experiments (full fine-tuning and LoRA).

---

## Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `LABEL_STUDIO_TOKEN` | API token from Label Studio (Account → Access Token) |
| `LABEL_STUDIO_URL` | Default: `http://localhost:8080` |
| `LABEL_STUDIO_PROJECT_ID` | Project ID in Label Studio (usually `1`) |
| `MLFLOW_TRACKING_URI` | Default: `http://localhost:5000` |
