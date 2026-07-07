# ClimaTag – Running Instructions

> Master's thesis – Damjan
> Faculty of Informatics and Digital Technologies, University of Rijeka
> Supervisor: Prof. dr. sc. Sanda Martinčić-Ipšić

These instructions apply to the **already configured machine** (WSL2 Ubuntu, repository
in `~/climate-nlp-platform`, conda env `climtag-env`). For setting up from scratch,
see [SETUP.md](SETUP.md).

---

## Quick start

The backend (which also serves the frontend) runs as a **systemd service** and starts
automatically. To start/check manually:

```bash
# 1. Docker services (Label Studio + MLflow) — if not already running
cd ~/climate-nlp-platform
docker compose -f docker/docker-compose.yml up -d

# 2. Backend service — check / start
sudo systemctl status climatag
sudo systemctl start climatag      # if not running
```

Open **http://localhost:8000** – the ClimaTag UI.

> If the Docker daemon is not running: `sudo systemctl start docker`
> (this machine uses the native Docker Engine inside Ubuntu, not Docker Desktop).

---

## Service mode

```bash
# Status
sudo systemctl status climatag

# Restart — required after: retraining (to load the new model)
# or rebuilding the frontend (npm run build)
sudo systemctl restart climatag

# Logs
journalctl -u climatag -n 50 --no-pager    # last 50 lines
journalctl -u climatag -f                   # follow live
```

---

## URLs

| Service | URL |
|---------|-----|
| ClimaTag application | http://localhost:8000 |
| FastAPI documentation | http://localhost:8000/docs |
| Label Studio | http://localhost:8080 |
| MLflow | http://localhost:5000 |

---

## Application pages

| Page | Description |
|------|-------------|
| **NER** | Named Entity Recognition – enter text, pick a model (Baseline or Climate Model), analyse entities |
| **Annotate** | Human-in-the-loop annotation – the model pre-annotates text, the user corrects it, corrections are pushed to Label Studio |
| **Train** | Fine-tuning of the GLiNER model on new annotations, with live progress tracking |
| **Experiments** | Overview of MLflow experiments and training metrics |
| **Classify** | Classification of scientific text into 20 SciDCC categories (SciClimateBERT) |

---

## NER models

| Model | Description |
|-------|-------------|
| **Baseline** | GLiNER medium-v2.1 – recognises the 28 climate entity categories (CliReNER set) |
| **Climate Model** | Fine-tuned via experience replay – adds the Climate Model category (CMIP6, ERA5, RCP/SSP scenarios...) on top of the existing 28 |

Climate Model results on the held-out test set:
**Precision 0.9744 · Recall 0.6129 · F1 0.7525**
(one-to-one overlap matching; methodology details in [TRAINING.md](TRAINING.md))

---

## Adding new annotations & retraining

1. **Annotate** page → enter text → correct the pre-annotations → Save
2. **Train** page → configure parameters → **Start training**
   (parameters are validated; a second concurrent run is rejected with 409)
3. Follow progress in the log panel
4. When finished: `sudo systemctl restart climatag` (loads the new model)

Run results are visible on the **Experiments** page and in MLflow.

---

## Conda environment

```bash
conda activate climtag-env

# GPU check
python -c "import torch; print(torch.cuda.is_available())"

# Tests (span conversion — regression suite)
python -m pytest tests/ -v
```

### Key packages

| Package | Version |
|---------|---------|
| Python | 3.10 |
| torch | 2.5.1+cu121 |
| gliner | latest |
| fastapi | 0.135.x |
| mlflow | 3.11.x |

---

## Known issues

**Docker permission denied**
```bash
sudo usermod -aG docker $USER
# close and reopen the terminal
```

**Service "running" but the UI is unreachable**
1. `curl http://127.0.0.1:8000/health` — if it fails, check `journalctl -u climatag -n 50`
2. `ls frontend/dist/index.html` — if missing: `cd frontend && npm run build`,
   then **restart the service** (the static mount registers at startup)

**Training failed (status "failed")**
The subprocess traceback is in the log panel / `/api/train/status`. After fixing the
cause: Reset on the Train page (or `POST /api/train/reset`) and start again — no
backend restart needed, each training run is a fresh subprocess.

**Backend can't find module 'backend'**
Run uvicorn/training from the project root (`~/climate-nlp-platform`), not from a
subdirectory. For the service: `WorkingDirectory` in the unit file must be the repo root.

**GLiNER slow without a GPU**
Normal – CPU inference is 10–20× slower. Check `nvidia-smi`.

**MLflow artifact PermissionError**
```bash
chmod 777 ~/climate-nlp-platform/docker/mlflow-artifacts
```

**WSL specifics** (default distro, interop, Vmmem, volume prefixes...)
See the Troubleshooting section in [SETUP.md](SETUP.md) — it covers all known WSL2 pitfalls.

---

*Last updated: July 2026*