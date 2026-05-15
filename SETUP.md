# Setup Guide

Step-by-step setup for ClimaTag on a fresh Linux or WSL2 machine.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the repository](#2-clone-the-repository)
3. [Python environment](#3-python-environment-conda)
4. [Download models](#4-download-models)
5. [Environment variables](#5-environment-variables)
6. [Docker services](#6-docker-services-label-studio--mlflow)
7. [Backend](#7-backend)
8. [Verify](#8-verify-everything-works)

---

## 1. Prerequisites

### Linux / WSL2 (Ubuntu 22.04)

```bash
sudo apt update && sudo apt install -y git curl wget
```

### Windows users → WSL2

Run all commands inside **WSL2 (Ubuntu 22.04)**. Install if needed:
```powershell
# PowerShell (Admin)
wsl --install
```

### CUDA (recommended)

Install NVIDIA drivers for your GPU. Verify GPU visibility inside WSL2:
```bash
nvidia-smi
```

### Docker

```bash
# Linux
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

Verify:
```bash
docker --version        # 24+
docker compose version  # 2.x
```

---

## 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/climate-nlp-platform.git
cd climate-nlp-platform
```

---

## 3. Python environment (conda)

### Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash && source ~/.bashrc
```

### Create environment

```bash
conda create -n climtag-env python=3.10 -y
conda activate climtag-env
```

### Install dependencies

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gliner
pip install fastapi uvicorn python-dotenv requests
pip install mlflow pandas pyarrow
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

---

## 4. Download models

Models are not in Git (too large). Download manually:

### GLiNER baseline

```bash
mkdir -p models/ner_gliner_baseline
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='urchade/gliner_medium-v2.1',
    local_dir='models/ner_gliner_baseline'
)
"
```

### GLiNER Climate Model (fine-tuned)

This model is produced by the fine-tuning pipeline. To create it:
1. Add Climate Model annotations in Label Studio (see [TRAINING.md](TRAINING.md))
2. Run fine-tuning via the **Train** tab in ClimaTag UI, or via CLI

Until fine-tuned, the platform falls back to the baseline model automatically.

### CliReNER SILVER dataset (for fine-tuning)

Required only for retraining. Place in `data/raw/CliReNER_SILVER/`:
- `train-00000-of-00001.parquet`
- `validation-00000-of-00001.parquet`
- `test-00000-of-00001.parquet`

---

## 5. Environment variables

```bash
cp .env.example .env
```

Edit `.env`:
```env
LABEL_STUDIO_TOKEN=<your_token>
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_PROJECT_ID=1
MLFLOW_TRACKING_URI=http://localhost:5000
ALLOWED_ORIGINS=http://localhost:8000
```

Get the Label Studio token after starting the service (step 6).

---

## 6. Docker services (Label Studio + MLflow)

### Create MLflow artifact directory

```bash
mkdir -p docker/mlflow-artifacts
chmod 777 docker/mlflow-artifacts
```

> Without this, MLflow artifact uploads will fail with a `PermissionError`.

### Start services

```bash
docker compose -f docker/docker-compose.yml up -d
```

Wait ~30 seconds, then verify:

| Service | URL |
|---|---|
| Label Studio | http://localhost:8080 |
| MLflow | http://localhost:5000 |

### Label Studio first-time setup

1. Open http://localhost:8080 and register an account
2. Go to **Account & Settings → Access Token** → copy the token
3. Paste it as `LABEL_STUDIO_TOKEN` in `.env`
4. Create a new project (e.g. "ClimaTag NER")
5. Add the NER labeling interface under **Settings → Labeling Interface**:

```xml
<View>
  <Text name="text" value="$text"/>
  <Labels name="label" toName="text">
    <Label value="Asset" background="#8B4513"/>
    <Label value="Chemical" background="#4169E1"/>
    <Label value="Climate Model" background="#22c55e"/>
    <Label value="Disease" background="#DC143C"/>
    <Label value="Ecosystem" background="#228B22"/>
    <Label value="Location" background="#FF8C00"/>
    <Label value="Organism" background="#9370DB"/>
    <Label value="Person" background="#708090"/>
    <Label value="Quantity" background="#20B2AA"/>
  </Labels>
</View>
```

---

## 7. Backend

The backend serves both the API and the React frontend (production build).

```bash
conda activate climtag-env
cd climate-nlp-platform
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

At startup you should see:
```
Loading GLiNER model from models/ner_gliner_climate_model...
GLiNER model ready! (active: climate_model)
```

Or if the fine-tuned model doesn't exist yet:
```
Loading GLiNER model from models/ner_gliner_baseline...
GLiNER model ready! (active: baseline)
```

### Run as a system service (optional, for production)

```bash
sudo cp climatag.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable climatag
sudo systemctl start climatag
```

---

## 8. Verify everything works

```
[ ] http://localhost:8000        – ClimaTag UI loads
[ ] http://localhost:8000/docs   – FastAPI docs load
[ ] http://localhost:8000/health – returns {"status": "ok"}
[ ] http://localhost:8080        – Label Studio login page
[ ] http://localhost:5000        – MLflow experiment list

[ ] NER page: paste a climate sentence → entities appear
[ ] Annotate page: pre-annotation works
[ ] Train page: annotation count shows
[ ] Experiments page: GLiNER runs visible
```

### Troubleshooting

**`ModuleNotFoundError: No module named 'backend'`**
Run uvicorn from the project root, not from inside `backend/`.

**`CUDA out of memory`**
GLiNER medium uses ~2GB VRAM. If you have less, remove `.to("cuda")` in `ner_service.py` to run on CPU (slower).

**Docker service logs**
```bash
docker compose -f docker/docker-compose.yml logs label-studio
docker compose -f docker/docker-compose.yml logs mlflow
```
