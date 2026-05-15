# ClimaTag – Upute za pokretanje

> Magistarski rad – Damjan  
> Fakultet informatike i digitalnih tehnologija, Rijeka  
> Mentorica: Prof. dr. sc. Sanda Martinčić-Ipšić

---

## Brzo pokretanje

```bash
# 1. Aktiviraj environment
conda activate climtag-env

# 2. Pokreni Docker servise (Label Studio + MLflow)
cd ~/climate-nlp-platform
docker compose -f docker/docker-compose.yml up -d

# 3. Pokreni backend (servira i frontend i API)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Otvori **http://localhost:8000** – ClimaTag UI.

---

## Servisni način rada (automatski start)

Backend je konfiguriran kao systemd servis i automatski se pokreće pri startu sustava:

```bash
# Status
sudo systemctl status climatag

# Restart (npr. nakon retraininga)
sudo systemctl restart climatag

# Logovi
sudo journalctl -u climatag -n 50 --no-pager
```

Docker servise (Label Studio, MLflow) treba pokrenuti ručno ako nisu već pokrenuti:
```bash
cd ~/climate-nlp-platform
docker compose -f docker/docker-compose.yml up -d
```

---

## URL-ovi

| Servis | URL |
|--------|-----|
| ClimaTag aplikacija | http://localhost:8000 |
| FastAPI dokumentacija | http://localhost:8000/docs |
| Label Studio | http://localhost:8080 |
| MLflow | http://localhost:5000 |

---

## Stranice aplikacije

| Stranica | Opis |
|----------|------|
| **NER** | Named Entity Recognition – unesi tekst, odaberi model (Baseline ili Climate Model), analiziraj entitete |
| **Annotate** | Human-in-the-loop anotacija – model pre-anotira tekst, ti ispravljaš, šalje se u Label Studio |
| **Train** | Fine-tuning GLiNER modela na novim anotacijama iz Label Studija |
| **Experiments** | Pregled MLflow eksperimenata i metrika treninga |

---

## NER modeli

| Model | Opis |
|-------|------|
| **Baseline** | GLiNER medium-v2.1 – pretreniran na općim NER datasetima, prepoznaje 28 klimatskih kategorija entiteta |
| **Climate Model** | Fine-tuned na Climate Model anotacijama (CMIP6, ERA5, SSP scenariji...) – dodaje kategoriju Climate Model uz 28 postojećih |

---

## Dodavanje novih anotacija i retraining

1. Idi na **Annotate** stranicu, unesi tekst, ispravi anotacije, klikni Save
2. Idi na **Train** stranicu, podesi parametre, klikni **Start training**
3. Prati napredak u log panelu
4. Nakon završetka: `sudo systemctl restart climatag`

---

## Conda environment

```bash
# Aktivacija
conda activate climtag-env

# Provjera GPU-a
python -c "import torch; print(torch.cuda.is_available())"
```

### Ključni paketi

| Paket | Verzija |
|-------|---------|
| Python | 3.10 |
| torch | 2.5.1+cu121 |
| gliner | latest |
| fastapi | 0.135.3 |
| mlflow | 3.11.1 |

---

## Poznati problemi

**Docker permission denied**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Backend ne vidi module 'backend'**
Pokreni uvicorn iz root direktorija projekta (`~/climate-nlp-platform`), ne iz poddirektorija.

**GLiNER spor bez GPU-a**
Normalno – CPU inference je 10-20x sporiji. Provjeri `nvidia-smi` i CUDA instalaciju.

**MLflow artifact PermissionError**
```bash
chmod 777 ~/climate-nlp-platform/docker/mlflow-artifacts
```

---

*Zadnje ažuriranje: svibanj 2026.*
