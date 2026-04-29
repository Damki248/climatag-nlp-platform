# Setup Guide

This guide walks through setting up the ClimaTag platform from scratch on a fresh machine (Linux or WSL2 on Windows).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the repository](#2-clone-the-repository)
3. [Python environment (conda)](#3-python-environment-conda)
4. [Download models](#4-download-models)
5. [Dataset setup](#5-dataset-setup)
6. [Environment variables](#6-environment-variables)
7. [Docker services](#7-docker-services-label-studio--mlflow)
8. [Backend](#8-backend)
9. [Frontend](#9-frontend)
10. [Verify everything works](#10-verify-everything-works)

---

## 1. Prerequisites

### Linux / WSL2 (Ubuntu 22.04)

Install system dependencies:

```bash
sudo apt update && sudo apt install -y git curl wget unzip
```

### Windows users → use WSL2

All commands in this guide run inside **WSL2 (Ubuntu 22.04)**. Do not run them in PowerShell or CMD.

Install WSL2 if you haven't already:
```powershell
# In PowerShell (Admin)
wsl --install
```

### CUDA (optional but recommended)

If you have an NVIDIA GPU, install CUDA 12.1 drivers on the host system. The conda environment will install the correct PyTorch build automatically.

To verify GPU is visible inside WSL2:
```bash
nvidia-smi
```

### Docker

Install Docker Desktop (Windows) with WSL2 integration enabled, or Docker Engine directly on Linux:

```bash
# Linux
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in, or run: newgrp docker
```

Verify:
```bash
docker --version       # 24+
docker compose version # 2.x
```

### Node.js (for frontend)

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version  # 20.x
```

---

## 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/climate-nlp-platform.git
cd climate-nlp-platform
```

---

## 3. Python environment (conda)

### Install Miniconda (if not installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash && source ~/.bashrc
```

### Create the environment

```bash
conda create -n spanmarker-env python=3.10 -y
conda activate spanmarker-env
```

### Install dependencies

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.50.0 datasets peft accelerate
pip install span_marker==1.7.0
pip install fastapi uvicorn python-dotenv
pip install mlflow evaluate scikit-learn
```

Verify GPU (if available):
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

---

## 4. Download models

Models are too large for Git and must be downloaded manually.

### NER model (CliReNER)

```bash
mkdir -p models/ner_baseline
```

Download from Hugging Face and place in `models/ner_baseline/`:

```bash
# Option A: using huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='P0L3/CliReNER-cliscibert_scivocab_uncased',
    local_dir='models/ner_baseline'
)
"
```

### Classification model (SciClimateBERT fine-tuned)

The fine-tuned classification model is not on Hugging Face – it is produced by the training script.

To train it from scratch, see [TRAINING.md](TRAINING.md).

If you have a pre-trained checkpoint, place it at:
```
models/cls_full_ft_best/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json
```

---

## 5. Dataset setup

### SciDCC (classification)

The SciDCC dataset (CSV) must be preprocessed before training or serving.

1. Place the raw `SciDCC.csv` in `data/raw/`
2. Run the preprocessing notebook:
   ```bash
   conda activate spanmarker-env
   jupyter notebook notebooks/02_SciDCC_Preprocessing.ipynb
   ```
   This creates:
   - `data/processed/hf_body/` – HuggingFace DatasetDict (body input)
   - `data/processed/hf_title_summary/` – HuggingFace DatasetDict (title+summary input)
   - `data/processed/label_map.json` – label ↔ id mapping
   - `data/processed/class_weights.json` – weights for imbalanced training

> The backend's classification service reads `data/processed/label_map.json` at startup. This file must exist before starting the backend.

---

## 6. Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
LABEL_STUDIO_TOKEN=<your_token>        # from Label Studio → Account → Access Token
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_PROJECT_ID=1
MLFLOW_TRACKING_URI=http://localhost:5000
```

You can get the Label Studio token after starting the service (step 7) and logging in.

---

## 7. Docker services (Label Studio + MLflow)

```bash
docker compose -f docker/docker-compose.yml up -d
```

Wait ~30 seconds for services to start, then verify:

| Service | URL | Expected |
|---|---|---|
| Label Studio | http://localhost:8080 | Login page |
| MLflow | http://localhost:5000 | Experiment list |

### Label Studio first-time setup

1. Open http://localhost:8080
2. Register an account (any email/password)
3. Go to **Account & Settings → Access Token** → copy the token
4. Paste it as `LABEL_STUDIO_TOKEN` in your `.env`
5. Create a new project (name it anything, e.g. "ClimaTag NER")
6. Note the project ID from the URL (`/projects/1/` → ID is `1`)
7. Update `LABEL_STUDIO_PROJECT_ID` in `.env` if different

### Stopping services

```bash
docker compose -f docker/docker-compose.yml down
```

Data is persisted in Docker volumes (`label_studio_data`, `mlflow_data`) and survives restarts.

---

## 8. Backend

```bash
conda activate spanmarker-env
cd ~/climate-nlp-platform   # or wherever you cloned
uvicorn backend.app.main:app --reload --port 8000
```

On startup, the backend loads both models into memory. With a GPU you'll see:
```
Loading NER model from models/ner_baseline...
NER model ready!
Loading classification model from models/cls_full_ft_best...
Classification model ready! (20 classes)
```

Verify at http://localhost:8000/health → `{"status": "ok", "version": "0.1.0"}`

Interactive API docs: http://localhost:8000/docs

### Common issues

**`ModuleNotFoundError: No module named 'backend'`**  
Make sure you run uvicorn from the project root (`~/climate-nlp-platform`), not from inside `backend/`.

**`CUDA out of memory`**  
Both models are loaded simultaneously. If you have less than 8GB VRAM, edit `ner_service.py` and `cls_service.py` to remove the `.cuda()` call – models will run on CPU (slower).

**`FileNotFoundError: models/cls_full_ft_best`**  
The classification model is not downloaded. See [step 4](#4-download-models) or [TRAINING.md](TRAINING.md).

---

## 9. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

The frontend expects the backend at `http://localhost:8000`. If you change the backend port, update the API base URL in `frontend/src/`.

---

## 10. Verify everything works

Run through this checklist:

```
[ ] http://localhost:5173     – ClimaTag UI loads
[ ] http://localhost:8000/docs – FastAPI docs load
[ ] http://localhost:8080     – Label Studio login page
[ ] http://localhost:5000     – MLflow experiment list

[ ] NER page: paste a climate sentence → entities appear
[ ] Classify page: paste text → top-3 categories appear
[ ] Annotate page: text loads with pre-annotated entities
```

If any service fails, check its logs:

```bash
# Backend logs – visible in the terminal where uvicorn is running

# Docker service logs
docker compose -f docker/docker-compose.yml logs label-studio
docker compose -f docker/docker-compose.yml logs mlflow
```
