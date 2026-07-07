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
git clone https://github.com/Damki248/climatag-nlp-platform.git
cd climatag-nlp-platform
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
conda create -n climatag-env python=3.10 -y
# You will be asked to accept the Terms of service first, so execute commands provided in the output, then re-run the command above
conda activate climatag-env
```

### Install dependencies

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

---

## 4. Download models

Models are not in Git (too large). The GLiNER **baseline** comes from Hugging Face;
the two **fine-tuned** models (GLiNER Climate Model + classifier) come from Google Drive.

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

### CliReNER SILVER dataset (for GLiNER fine-tuning)

Required only for retraining. Place in `data/raw/CliReNER_SILVER/`:
- `train-00000-of-00001.parquet`
- `validation-00000-of-00001.parquet`
- `test-00000-of-00001.parquet`

---

### Fine-tuned models — GLiNER Climate Model + classifier (Google Drive, temporary)

The two fine-tuned models are not in Git. They are temporarily hosted on Google
Drive as two separate archives. Download and unzip both into `models/`.

```bash
# 0. Install gdown (Google Drive downloader)
pip install gdown
mkdir -p models

# 1. Fine-tuned GLiNER Climate Model
#    Replace FILE_ID_GLINER with the id from its share link
gdown "https://drive.google.com/file/d/FILE_ID_GLINER/view" -O gliner_climate_model.zip
unzip gliner_climate_model.zip -d models/
rm gliner_climate_model.zip

# 2. Classification model (SciClimateBERT)
#    Replace FILE_ID_CLS with the id from its share link
gdown "https://drive.google.com/file/d/FILE_ID_CLS/view" -O cls_model.zip
unzip cls_model.zip -d models/
rm cls_model.zip
```

After unzipping, `models/` must contain:

```
models/
├── ner_gliner_baseline/                    # from Hugging Face (previous step)
├── ner_gliner_climate_model/               # from Drive (archive 1)
│   ├── gliner_config.json
│   └── pytorch_model.bin
└── cls_full_ft_body_lr2e5_batch16_exp07/
    └── checkpoint-11110/                    # from Drive (archive 2)
        ├── config.json
        ├── model.safetensors
        └── tokenizer files...
```

> The Drive links are temporary (for thesis review only).

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
    <Label value="Asset" background="#B91C1C"/>
    <Label value="Body Part" background="#EA580C"/>
    <Label value="Body of Water" background="#2563EB"/>
    <Label value="Chemical" background="#7C3AED"/>
    <Label value="Disease" background="#9F1239"/>
    <Label value="Ecosystem" background="#16A34A"/>
    <Label value="Energy Source" background="#CA8A04"/>
    <Label value="Field of Study" background="#4F46E5"/>
    <Label value="Geographical Feature" background="#8B5CF6"/>
    <Label value="Intellectual Artefact" background="#0891B2"/>
    <Label value="Location" background="#0E7490"/>
    <Label value="Mathematical Expression" background="#C026D3"/>
    <Label value="Measuring Device" background="#65A30D"/>
    <Label value="Meteorological Phenomenon" background="#0284C7"/>
    <Label value="Method" background="#D97706"/>
    <Label value="Natural Disaster" background="#7F1D1D"/>
    <Label value="Natural Phenomenon" background="#0D9488"/>
    <Label value="Organism" background="#15803D"/>
    <Label value="Organization" background="#1D4ED8"/>
    <Label value="Other" background="#6B7280"/>
    <Label value="Person" background="#6D28D9"/>
    <Label value="Physical Artefact" background="#C2410C"/>
    <Label value="Physical Phenomenon" background="#0F766E"/>
    <Label value="Policy" background="#9A3412"/>
    <Label value="Quantity" background="#A16207"/>
    <Label value="Satellite" background="#4B5563"/>
    <Label value="System" background="#1E40AF"/>
    <Label value="Time Period" background="#B45309"/>
    <Label value="Climate Model" background="#22C55E"/>
  </Labels>
</View>
```

---

## 7. Backend

The backend serves both the API and the React frontend (production build).

```bash
conda activate climtag-env
cd climatag-nlp-platform
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

Before executing these commands, open the `climatag.service` file and chage every `<your_username>` placeholder with the system user.

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
[ ] Classify page: paste an abstract → predictions appear
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
