# Setup Guide

Step-by-step setup for ClimaTag on a fresh Linux or WSL2 machine.

The guide assumes no prior tooling — every prerequisite is installed from scratch. Commands were validated on a clean WSL2 Ubuntu installation; where WSL2 behaves differently from a native Linux server, the difference is called out explicitly.

---

## Table of Contents

1. [WSL2 prerequisites (Windows users)](#1-wsl2-prerequisites-windows-users)
2. [System packages & Docker Engine](#2-system-packages--docker-engine)
3. [Clone the repository](#3-clone-the-repository)
4. [Python environment (conda)](#4-python-environment-conda)
5. [Node.js & frontend build](#5-nodejs--frontend-build)
6. [Download models & data](#6-download-models--data)
7. [Environment variables](#7-environment-variables)
8. [Docker services (Label Studio + MLflow)](#8-docker-services-label-studio--mlflow)
9. [Backend](#9-backend)
10. [Verify everything works](#10-verify-everything-works)
11. [Troubleshooting](#11-troubleshooting)

Native Linux users can skip section 1 and start at section 2.

---

## 1. WSL2 prerequisites (Windows users)

### Install / update WSL2

```powershell
# PowerShell (Admin)
wsl --install -d Ubuntu-24.04
wsl --update
```

Ubuntu 22.04 also works. After installation, confirm you are on WSL **2**:

```powershell
wsl -l -v      # VERSION column must say 2; the * (default) must be on Ubuntu
```

### Rule 1: work in the Linux filesystem

All project files must live in the Linux home directory (`~`), **never** under `/mnt/c/...`.
The Windows-mounted filesystem is drastically slower and causes permission and file-watching
problems with Docker, npm, and model loading.

### Rule 2: enable systemd (needed for the climatag.service unit and Docker auto-start)

```bash
# inside Ubuntu
sudo tee /etc/wsl.conf > /dev/null <<'EOF'
[boot]
systemd=true
EOF
```

Then from PowerShell: `wsl --shutdown`, wait ~10 s, and reopen Ubuntu.
Verify with `systemctl list-units --type=service | head` (should list services, not error out).

### Rule 3: cap WSL2 resources

Training + Docker + Windows can exhaust RAM and hard-crash the WSL VM. Create
`C:\Users\<you>\.wslconfig` (Windows side, plain text):

```ini
[wsl2]
memory=12GB
processors=6
swap=16GB
```

Adjust to your machine — leave Windows at least 4–6 GB. A generous swap means heavy
training slows down instead of killing the VM. Apply with `wsl --shutdown`.

### Rule 4: do NOT use Docker Desktop

Install the native Docker Engine **inside Ubuntu** (section 2). Docker Desktop's WSL
integration adds its own `docker-desktop` distributions that can hijack the default
distro, interfere with WSL shutdown, and caused repeated VM hangs during development
of this project. The native Engine is simpler and stable.

### Recommended: GPU check

Install the NVIDIA driver on **Windows** (no driver install inside WSL needed), then verify inside Ubuntu:

```bash
nvidia-smi
```

### Recommended: full backup ability

At any point you can snapshot the entire distribution from PowerShell:

```powershell
wsl --export Ubuntu-24.04 D:\wsl-backup\ubuntu-backup.tar
```

Cheap insurance before risky changes.

---

## 2. System packages & Docker Engine

```bash
sudo apt update && sudo apt install -y git curl wget unzip
```

### Docker Engine (native)

```bash
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
```

**Close and reopen the terminal** (group membership loads on a new session), then verify:

```bash
docker --version         # 24+
docker compose version   # v2.x
docker ps                # empty table, no permission error
```

Start the daemon:

```bash
# with systemd (native Linux, or WSL with systemd enabled — see section 1):
sudo systemctl enable --now docker

# without systemd (start manually each session):
sudo service docker start
```

> Alternative: the official Docker repository (`curl -fsSL https://get.docker.com | sh`)
> gives newer versions; the Ubuntu packages above are sufficient for this project.

---

## 3. Clone the repository

```bash
cd ~                     # Linux home — see section 1, Rule 1
git clone https://github.com/Damki248/climatag-nlp-platform.git
cd climatag-nlp-platform
```

All subsequent commands run from the repository root unless stated otherwise.

---

## 4. Python environment (conda)

### Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash && source ~/.bashrc
```

### Create the environment

```bash
conda create -n climatag-env python=3.10 -y
# If asked to accept the Terms of Service, run the commands shown in the output,
# then re-run the create command.
conda activate climatag-env
```

### Install dependencies

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify GPU visibility:

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True   (False = CPU-only mode; everything works, just slower)
```

---

## 5. Node.js & frontend build

The FastAPI backend serves the **production build** of the React frontend from
`frontend/dist/`. Without this step the UI will not load — you will only get the API.

### Install Node.js via nvm (Linux-native)

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc
nvm install --lts
```

**WSL2 check:** confirm you are using Linux Node, not a Windows installation leaking
through `PATH`:

```bash
which node    # must be under ~/.nvm/...  — NOT /mnt/c/...
```

If it points to `/mnt/c/...`, the Windows Node will fail with `Exec format error` or
produce broken builds. Fix: ensure nvm's lines are at the end of `~/.bashrc` and reopen
the terminal.

### Build the frontend

```bash
cd frontend
npm install
npm run build          # output: frontend/dist/
cd ..
```

> Rebuild (`npm run build`) after any frontend change, then **restart the backend** —
> the static mount is registered once at startup.

---

## 6. Download models & data

Models are not in Git (too large). The GLiNER **baseline** comes from Hugging Face;
the two **fine-tuned** models come from Google Drive (temporary hosting for thesis review).

### 6.1 GLiNER baseline (Hugging Face)

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

### 6.2 Fine-tuned models (Google Drive)

```bash
pip install -U gdown        # -U is required: --fuzzy needs a recent version
mkdir -p models

# 1) Fine-tuned GLiNER Climate Model — replace FILE_ID_GLINER with the id from the share link
gdown --fuzzy "https://drive.google.com/file/d/FILE_ID_GLINER/view" -O gliner_climate_model.zip
unzip gliner_climate_model.zip -d models/
rm gliner_climate_model.zip

# 2) Classification model (SciClimateBERT) — replace FILE_ID_CLS with the id from the share link
gdown --fuzzy "https://drive.google.com/file/d/FILE_ID_CLS/view" -O cls_model.zip
unzip cls_model.zip -d models/
rm cls_model.zip
```

> `--fuzzy` lets gdown accept full `/file/d/<id>/view` share links. Without it (or with
> an old gdown), the download fails or saves an HTML page instead of the archive.

After unzipping, `models/` must contain:

```
models/
├── ner_gliner_baseline/                    # from Hugging Face (step 6.1)
├── ner_gliner_climate_model/               # from Drive (archive 1)
│   ├── gliner_config.json
│   └── pytorch_model.bin
└── cls_full_ft_body_lr2e5_batch16_exp07/
    └── checkpoint-11110/                   # from Drive (archive 2)
        ├── config.json
        ├── model.safetensors
        └── tokenizer files...
```

If the fine-tuned GLiNER model is missing, the backend automatically falls back to the
baseline. If the classification model is missing, the Classify page reports the service
as unavailable — the rest of the platform works normally.

### 6.3 CliReNER SILVER dataset (only for retraining)

Required only if you plan to fine-tune the NER model. Place the parquet files in
`data/raw/CliReNER_SILVER/`:

- `train-00000-of-00001.parquet`
- `validation-00000-of-00001.parquet`
- `test-00000-of-00001.parquet`

---

## 7. Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
LABEL_STUDIO_TOKEN=<your_token>          # obtained in step 8 — leave placeholder for now
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_PROJECT_ID=1
MLFLOW_TRACKING_URI=http://localhost:5000
ALLOWED_ORIGINS=http://localhost:8000
```

Optional: `CLS_MODEL_PATH` overrides the classification checkpoint path (defaults to the
path shown in section 6.2).

---

## 8. Docker services (Label Studio + MLflow)

### Create the MLflow artifact directory

```bash
mkdir -p docker/mlflow-artifacts
chmod 777 docker/mlflow-artifacts
```

> Without this, MLflow artifact uploads fail with a `PermissionError`.

### Start services

```bash
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml ps    # both containers "Up"
```

Wait ~30 seconds, then verify:

| Service | URL |
|---|---|
| Label Studio | http://localhost:8080 |
| MLflow | http://localhost:5000 |

> **Note on data location:** Label Studio's database and MLflow's run store live in named
> Docker **volumes**. The volume names are prefixed with the Compose project name — see
> the [Troubleshooting](#11-troubleshooting) entry "Label Studio account/projects
> disappeared" before moving or renaming anything.

### Label Studio first-time setup

1. Open http://localhost:8080 and register an account
2. Go to **Account & Settings → Access Token** → copy the token
3. Paste it as `LABEL_STUDIO_TOKEN` in `.env`
4. Create a new project (e.g. "ClimaTag NER"); confirm its ID matches `LABEL_STUDIO_PROJECT_ID`
   (the ID is in the project URL: `/projects/<ID>/...`)
5. Add the NER labeling interface under **Settings → Labeling Interface → Code** —
   this is the full 29-label configuration (28 CliReNER categories + Climate Model),
   matching the canonical set in `ner_labels.py`:

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

## 9. Backend

The backend serves both the API and the React frontend build (section 5).

### Option A — manual (recommended for first run and for development)

```bash
conda activate climatag-env
cd ~/climatag-nlp-platform          # must run from the repo root
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

At startup you should see:

```
Loading GLiNER model from models/ner_gliner_climate_model...
GLiNER model ready! (active: climate_model)
```

(or `baseline` if the fine-tuned model is not present).

> `--host 127.0.0.1` keeps the API local-only. Use `--host 0.0.0.0` only if the machine
> should be reachable from the network, and be aware the API has no authentication.

### Option B — systemd service (auto-start, production-style)

Requires systemd (native Linux, or WSL with systemd enabled — section 1, Rule 2).

1. Open `climatag.service` and replace every `<your_username>` placeholder with your
   system user. Double-check that `WorkingDirectory` points at the actual repo path
   and `ExecStart` at the actual conda env path — a mismatch is the most common cause
   of "service running but UI unreachable".
2. Install and start:

```bash
sudo cp climatag.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now climatag
```

Day-to-day commands:

```bash
sudo systemctl status climatag                     # is it up?
sudo systemctl restart climatag                    # after frontend rebuild or retraining
journalctl -u climatag -n 50 --no-pager            # last 50 log lines
journalctl -u climatag -f                          # follow logs live
```

Docker services (Label Studio, MLflow) are independent of this unit — start them
separately (section 8) or enable the Docker daemon at boot (section 2).

---

## 10. Verify everything works

```
[ ] http://localhost:8000        – ClimaTag UI loads
[ ] http://localhost:8000/docs   – FastAPI docs load
[ ] http://localhost:8000/health – returns {"status": "ok"}
[ ] http://localhost:8080        – Label Studio login page
[ ] http://localhost:5000        – MLflow experiment list

[ ] NER page: paste a climate sentence → entities appear
[ ]   ...including e.g. "RCP 8.5" recognised as Climate Model
[ ] Annotate page: pre-annotation works; Save pushes the task to Label Studio
[ ] Train page: annotation count shows; invalid parameters (e.g. epochs=5000) are rejected with 422
[ ] Experiments page: GLiNER runs visible
[ ] Classify page: paste an abstract → top-k predictions appear
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

---

## 11. Troubleshooting

### Backend & training

**`ModuleNotFoundError: No module named 'backend'` (or `'ner_labels'`, `'training'`)**
Run uvicorn / training from the project root, not from a subdirectory. For the systemd
service, this means `WorkingDirectory` must be the repo root.

**Service is "active (running)" but the UI is unreachable**
`systemctl status` only means uvicorn started. Check in order:
1. `curl http://127.0.0.1:8000/health` — if this fails, read `journalctl -u climatag -n 50`
2. `ls frontend/dist/index.html` — if missing, build the frontend (section 5) **and restart the service** (the static mount registers at startup)
3. Placeholders in `climatag.service` — wrong `WorkingDirectory` silently breaks both imports and frontend serving

**`/api/train/status` shows "failed" with a traceback in `logs`**
That is the training subprocess's own error — read the last log lines for the cause
(missing SILVER parquet, missing baseline model, MLflow unreachable). Fix, then
`POST /api/train/reset` and start again. No backend restart needed — each training run
is a fresh subprocess.

**Training status stuck on "running" after a crash/restart**
The status tracker stores the training PID and marks dead processes as "failed"
automatically on the next status read. If you ever hit a stuck state anyway, delete
`training_status.json` in the repo root.

**`CUDA out of memory`**
GLiNER medium uses ~2 GB VRAM; training needs more. Reduce `--batch` (or the batch
parameter in the Train UI). For CPU-only inference, remove `.to("cuda")` in
`ner_service.py`.

**MLflow artifact `PermissionError`**
```bash
chmod 777 docker/mlflow-artifacts
```

### Docker

**`permission denied` on `docker ps`**
```bash
sudo usermod -aG docker $USER
# then close and reopen the terminal
```

**Label Studio account/projects (or MLflow runs) disappeared**
Almost always a Compose **project-name prefix** issue, not data loss. Compose derives
volume names from the project name; running compose from a different directory or
after a rename creates *new, empty* volumes. Diagnose:
```bash
docker volume ls          # look for two pairs of similarly-named volumes
docker inspect <container> --format '{{range .Mounts}}{{.Name}} -> {{.Destination}}{{"\n"}}{{end}}'
```
Copy the data from the old volume into the one the container actually mounts:
```bash
docker compose -f docker/docker-compose.yml down
docker run --rm -v <OLD_VOLUME>:/from -v <NEW_VOLUME>:/to alpine sh -c "cp -a /from/. /to/"
docker compose -f docker/docker-compose.yml up -d
```
To prevent it permanently, pin the project name at the top of `docker/docker-compose.yml`
with a `name:` key.

**Container logs**
```bash
docker compose -f docker/docker-compose.yml logs label-studio
docker compose -f docker/docker-compose.yml logs mlflow
```

### Model downloads

**gdown downloads an HTML file / fails on the share link**
Update gdown and use `--fuzzy` (section 6.2): `pip install -U gdown`. Old versions
cannot parse `/file/d/<id>/view` links.

### WSL2-specific

**`wsl` opens a wrong/empty environment, or Ubuntu "won't start"**
Check `wsl -l -v` from PowerShell — the default (`*`) must be on Ubuntu, **not**
`docker-desktop` (a leftover from a Docker Desktop installation). Fix:
`wsl --set-default Ubuntu-24.04`. Start explicitly with `wsl -d Ubuntu-24.04` meanwhile.

**`Exec format error` when running any Windows .exe from WSL (including `code .`)**
Windows interop is broken, typically by systemd's binfmt handling. Permanent fix:
```bash
sudo mkdir -p /usr/lib/binfmt.d
echo ':WSLInterop:M::MZ::/init:PF' | sudo tee /usr/lib/binfmt.d/WSLInterop.conf
sudo systemctl restart systemd-binfmt
```

**WSL VM hangs, `wsl --shutdown` takes forever, `Vmmem` eats RAM**
Usually memory pressure or a stale Docker Desktop installation. Set limits in
`.wslconfig` (section 1, Rule 3), make sure Docker Desktop is fully uninstalled
(including its `docker-desktop` distributions), and run `wsl --update`. Do **not**
kill `Vmmem` from Task Manager — reboot Windows instead (killing it can corrupt the
distribution's virtual disk).

**`npm run build` fails strangely / `which node` shows `/mnt/c/...`**
Windows Node is leaking through PATH. Install Linux Node via nvm (section 5) and
reopen the terminal.

**General WSL instability**
`wsl --update` fixes a long tail of known bugs. Take a `wsl --export` backup
(section 1) before major changes.
