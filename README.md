# ClimaTag – Climate NLP Platform

An end-to-end web platform for **Named Entity Recognition (NER)** in the climate science domain, featuring human-in-the-loop annotation, in-app model fine-tuning, and experiment tracking.

Built as part of a Master's thesis at the Faculty of Informatics and Digital Technologies, University of Rijeka.

> **Supervisor:** Prof. dr. sc. Sanda Martinčić-Ipšić

---

## What the platform does

- **NER** – Recognises 28 climate-domain entity categories (chemicals, locations, organisms, quantities, etc.) plus a custom **Climate Model** category (CMIP6, ERA5, RCP/SSP scenarios, etc.) using GLiNER. Switch between the baseline and fine-tuned model at runtime.
- **Human-in-the-loop annotation** – The model pre-annotates text; the user corrects it in the ClimaTag UI; corrections are pushed to Label Studio for storage.
- **In-app fine-tuning** – Retrain the GLiNER model on new annotations directly from the UI, with live progress tracking, parameter validation, and automatic crash recovery of the training status.
- **Experiment tracking** – All training runs are logged to MLflow (hyperparameters, dataset sizes, Climate Model precision/recall/F1).
- **Text classification** – Classifies scientific climate texts into 20 topic categories (SciDCC dataset) using a fine-tuned SciClimateBERT model.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│           React Frontend (ClimaTag)                        │
│   NER  │  Annotate  │  Experiments  │  Train  │  Classify  │
└──────────────────────┬─────────────────────────────────────┘
                       │ HTTP (port 8000)
┌──────────────────────▼─────────────────────────────────────┐
│              FastAPI Backend                               │
│  /api/ner  │  /api/annotation  │  /api/train  │  /api/cls  │
└──────┬──────────────────────────────────────────┬──────────┘
       │                                          │
┌──────▼───────────────────────┐   ┌──────────────▼──────────┐
│      GLiNER NER model        │   │  SciClimateBERT         │
│  baseline / climate_model    │   │  classifier (20 classes)│
└──────┬───────────────────────┘   └─────────────────────────┘
       │
┌──────▼───────────────────────────────────────┐
│           Docker services                    │
│   Label Studio :8080  │  MLflow :5000        │
└──────────────────────────────────────────────┘
```

---

## Requirements

| Requirement | Version |
|---|---|
| OS | Linux or WSL2 (Ubuntu 22.04 / 24.04) |
| Python | 3.10 (via conda) |
| Node.js | 20 LTS+ (via nvm, for building the frontend) |
| CUDA | 12.1 (GPU recommended; CPU works but is slow) |
| Docker | Docker Engine 24+ with Compose v2 |

> **Windows users:** run everything inside WSL2, in the Linux filesystem (`~`), with the native
> Docker Engine installed **inside Ubuntu** — not Docker Desktop. See [SETUP.md](SETUP.md) for details and rationale.

---

## Project structure

```
climatag-nlp-platform/
├── backend/
│   └── app/
│       ├── api/               # FastAPI routers (ner, annotation, train, cls)
│       ├── services/          # Business logic (ner_service, label_studio_service, cls_service)
│       └── main.py            # App entry point + static file serving
├── frontend/                  # React + Vite (ClimaTag UI)
│   ├── src/constants/nerLabels.js   # JS mirror of the canonical label set
│   └── dist/                  # Production build (served by FastAPI, created by `npm run build`)
├── training/
│   ├── ner_gliner/train.py    # GLiNER fine-tuning (experience replay)
│   └── classification/train.py # SciClimateBERT fine-tuning (full FT / LoRA)
├── tests/                     # pytest tests (span conversion regression suite)
├── docker/
│   └── docker-compose.yml     # Label Studio + MLflow containers
├── data/
│   ├── raw/CliReNER_SILVER/   # SILVER NER dataset (parquet, not in Git)
│   └── annotations/           # Label Studio exports
├── models/                    # Model weights (not in Git – too large)
├── ner_labels.py              # Canonical NER label set (single source of truth)
├── climatag.service           # systemd unit for the backend
├── .env.example               # Environment variable template
└── README.md
```

> `models/` weights and the SILVER dataset are excluded from Git. See [SETUP.md](SETUP.md) for download instructions.

---

## Quick start

Full setup instructions: **[SETUP.md](SETUP.md)**. After setup, the platform runs on a single port:

```bash
# Start Docker services (Label Studio + MLflow)
docker compose -f docker/docker-compose.yml up -d

# Start backend (serves the frontend build + API)
conda activate climatag-env
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# — or, if installed as a systemd service:
sudo systemctl start climatag
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
| `LABEL_STUDIO_TOKEN` | API token from Label Studio (Account & Settings → Access Token) |
| `LABEL_STUDIO_URL` | Default: `http://localhost:8080` |
| `LABEL_STUDIO_PROJECT_ID` | Project ID in Label Studio (usually `1`) |
| `MLFLOW_TRACKING_URI` | Default: `http://localhost:5000` |
| `ALLOWED_ORIGINS` | CORS origins, comma-separated. Default: `http://localhost:5173` |
| `CLS_MODEL_PATH` | Optional. Path to the classification model checkpoint (has a sensible default) |

---

## Training

See **[TRAINING.md](TRAINING.md)** for GLiNER fine-tuning (in-app or CLI) and SciClimateBERT classification training.
Both fine-tuned models ship pre-trained — retraining is optional.

For running the already-configured thesis machine, see **[INSTRUCTIONS.md](INSTRUCTIONS.md)** (in Croatian).
