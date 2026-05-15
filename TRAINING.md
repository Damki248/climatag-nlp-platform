# Training Guide

This guide covers fine-tuning the GLiNER model on new Climate Model annotations.

---

## Overview

ClimaTag uses **GLiNER** (Generalist and Lightweight NER) for named entity recognition. The fine-tuning strategy uses **experience replay**: a mix of the CliReNER SILVER dataset (28 existing categories) and new Climate Model annotations, preventing catastrophic forgetting of old categories while learning the new one.

The production model (`models/ner_gliner_climate_model`) was fine-tuned with:
- 290 Climate Model annotations
- 803 SILVER dataset samples (silver_ratio=5:1 × 290 ≈ 1450, capped at available)
- 10 epochs, LR=5e-6, batch=8
- Results: **Climate Model F1 = 93.0%** (Precision 97.2%, Recall 89.1%)

---

## Option A – In-app fine-tuning (recommended)

1. Open ClimaTag at http://localhost:8000
2. Go to the **Annotate** tab and add new Climate Model annotations
3. Go to the **Train** tab
4. Configure parameters and click **Start training**
5. Monitor progress in the live log panel
6. When finished, restart the backend to load the new model:
   ```bash
   sudo systemctl restart climatag
   # or if running manually: Ctrl+C and restart uvicorn
   ```

---

## Option B – CLI fine-tuning

```bash
conda activate climtag-env
cd climate-nlp-platform

python -m training.ner_gliner.train \
  --cm_annotations  data/annotations/climate_model_annotations.json \
  --base_model      models/ner_gliner_baseline \
  --output_model    models/ner_gliner_climate_model \
  --epochs          10 \
  --lr              5e-6 \
  --batch           8 \
  --silver_ratio    5 \
  --experiment      climtag_ner_gliner \
  --run_name        gliner_v2
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--cm_annotations` | `data/annotations/climate_model_annotations.json` | Label Studio export with Climate Model annotations |
| `--base_model` | `models/ner_gliner_baseline` | Starting model |
| `--output_model` | `models/ner_gliner_climate_model` | Where to save the fine-tuned model |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `5e-6` | Learning rate |
| `--batch` | `8` | Batch size |
| `--silver_ratio` | `5` | SILVER:Climate Model sample ratio |
| `--val_split` | `0.15` | Validation set fraction |

---

## MLflow tracking

Every run logs to MLflow at http://localhost:5000 under experiment `climtag_ner_gliner`.

**Parameters logged:** base_model, epochs, lr, batch, silver_ratio, train_size, val_size, cm_samples, seed

**Metrics logged:**
- `cm_precision` – Climate Model precision
- `cm_recall` – Climate Model recall
- `cm_f1` – Climate Model F1

View results in the **Experiments** tab of ClimaTag or directly at http://localhost:5000.

---

## Adding annotations

To add new Climate Model annotations for the next training run:

1. Go to **Annotate** tab in ClimaTag – enter text and pre-annotate with the current model
2. Correct any mistakes in the annotation editor
3. Submit – corrections are sent to Label Studio
4. Export annotations from Label Studio:
   ```bash
   curl -s -H "Authorization: Token YOUR_TOKEN" \
     "http://localhost:8080/api/projects/1/export?exportType=JSON" \
     -o data/annotations/climate_model_annotations.json
   ```
5. Run fine-tuning (Option A or B above)

---

## Hardware requirements

| Setup | Training time (10 epochs, ~1000 samples) |
|---|---|
| RTX 3050 8GB, batch=8 | ~5–8 minutes |
| CPU only | ~30–60 minutes |

If you run out of GPU memory, reduce `--batch` to `4`.
