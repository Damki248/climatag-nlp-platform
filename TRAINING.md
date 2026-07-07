# Training Guide

This guide covers fine-tuning both models: the **GLiNER NER model** (adding the Climate Model
category) and the **SciClimateBERT classifier** (SciDCC topic classification).

---

## 1. GLiNER fine-tuning (NER)

### Overview

ClimaTag uses **GLiNER** (Generalist and Lightweight NER). The fine-tuning strategy is
**experience replay**: new Climate Model annotations are mixed with samples from the
CliReNER SILVER dataset (the 28 existing categories) at a configurable ratio, which
teaches the new category while preventing catastrophic forgetting of the old ones.

Pipeline details (implemented in `training/ner_gliner/train.py`):

- **Label set** — imported from `ner_labels.py`, the canonical single source of truth
  (28 CliReNER categories + Climate Model). The IOB2 index mapping for the SILVER
  dataset is derived from the same list, so the category order in `ner_labels.py`
  is a contract with the dataset and must not be reordered.
- **Span conversion** — Label Studio character spans are mapped to whitespace-token
  spans with overlap-based matching: annotations that cut through a token (e.g.
  "CMIP6" inside "(CMIP6)", or multi-token forms like "RCP 8.5") expand to the full
  covering tokens; degenerate spans are logged and skipped. This logic is covered by
  the regression tests in `tests/test_span_conversion.py`.
- **Held-out test split** — 20 % of Climate Model samples (min 10) are set aside
  **before** the replay mix is built; they never enter training or validation.
- **Model selection** — the best epoch by validation loss is kept
  (`load_best_model_at_end=True`), not the last epoch.
- **Reproducibility** — a single `set_seed(seed)` call seeds Python, NumPy and
  PyTorch (default seed 42); all sizes and hyperparameters are logged to MLflow.

### Evaluation methodology

The Climate Model category is evaluated on the held-out test set with
**overlap-based, one-to-one matching**: a prediction counts as a true positive if it
overlaps a ground-truth span with the correct label, and each ground-truth span can be
matched by **at most one** prediction (additional overlapping predictions count as
false positives). Overlap matching is used because annotators are not perfectly
consistent about span borders (e.g. "CMIP6" vs "CMIP6 models"); the one-to-one
constraint keeps precision honest.

### Current production model

The shipped model (`models/ner_gliner_climate_model`) was fine-tuned with
~290 Climate Model annotations, silver_ratio 5:1, 10 epochs, LR 5e-6, batch 8.

Held-out Climate Model results:

| Metric | Value |
|---|---|
| Precision | 0.9744 |
| Recall | 0.6129 |
| F1 | 0.7525 |

The model is deliberately conservative: when it predicts Climate Model it is almost
always right, which suits its pre-annotation role (few false suggestions for the
annotator to delete), at the cost of missing some mentions.

> Results reported by earlier versions of this document (F1 ≈ 0.93) were produced by a
> previous pipeline with a span-conversion bug and a more permissive evaluation
> (many-to-one matching, last-epoch model). They are not comparable to the numbers above.

### Option A – In-app fine-tuning (recommended)

1. Open ClimaTag at http://localhost:8000
2. **Annotate** tab → add new Climate Model annotations (they are stored in Label Studio)
3. **Train** tab → configure parameters → **Start training**
4. Monitor progress in the live log panel (`/api/train/status` under the hood)
5. When finished, restart the backend to load the new model:
   ```bash
   sudo systemctl restart climatag
   # or, if running manually: Ctrl+C and restart uvicorn
   ```

The training API validates parameters and rejects out-of-range values with HTTP 422:

| Parameter | Range | Default |
|---|---|---|
| epochs | 1–50 | 10 |
| lr | (0, 1e-2] | 5e-6 |
| batch | 1–64 | 8 |
| silver_ratio | 1–20 | 5 |
| run_name | ≤100 chars, `[\w\-.]` only | auto-generated |

Only one training run can be active at a time (a second start returns HTTP 409). The
status tracker records the subprocess PID; if the backend or the training process dies,
the status is automatically marked *failed* on the next read and can be cleared with
`POST /api/train/reset`.

### Option B – CLI fine-tuning

```bash
conda activate climatag-env
cd ~/climatag-nlp-platform

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

| Argument | Default | Description |
|---|---|---|
| `--cm_annotations` | `data/annotations/climate_model_annotations.json` | Label Studio export with Climate Model annotations |
| `--silver_train` | `data/raw/CliReNER_SILVER/train-....parquet` | SILVER dataset (see SETUP.md §6.3) |
| `--base_model` | `models/ner_gliner_baseline` | Starting model |
| `--output_model` | `models/ner_gliner_climate_model` | Where the fine-tuned model is saved |
| `--epochs` | `10` | Training epochs |
| `--lr` | `5e-6` | Learning rate |
| `--batch` | `8` | Batch size |
| `--silver_ratio` | `5` | SILVER : Climate Model sample ratio |
| `--val_split` | `0.15` | Validation fraction of the mixed set |
| `--seed` | `42` | Random seed (Python/NumPy/PyTorch) |

### Adding annotations for the next run

1. **Annotate** tab → enter text → pre-annotate with the current model → correct → Save
   (corrections are pushed to Label Studio)
2. Export annotations from Label Studio:
   ```bash
   curl -s -H "Authorization: Token YOUR_TOKEN" \
     "http://localhost:8080/api/projects/1/export?exportType=JSON" \
     -o data/annotations/climate_model_annotations.json
   ```
3. Run fine-tuning (Option A or B)

### MLflow tracking

Every run logs to MLflow (http://localhost:5000) under experiment `climtag_ner_gliner`.

- **Parameters:** base_model, epochs, lr, batch, silver_ratio, train_size, val_size,
  cm_train, cm_test, seed
- **Metrics:** `cm_precision`, `cm_recall`, `cm_f1` (held-out Climate Model evaluation)

Results are also visible in the **Experiments** tab of ClimaTag.

---

## 2. SciClimateBERT classification training

The classifier (`training/classification/train.py`) fine-tunes
[`P0L3/SciClimateBERT`](https://huggingface.co/P0L3/SciClimateBERT) on the SciDCC
dataset (20 topic categories). The shipped checkpoint was trained with full fine-tuning
on the article body; retraining is optional.

Key features:

- **Strategies:** `--strategy full_ft` (default) or `--strategy lora` (PEFT; adapters
  are merged into the base weights on save, so the result loads like any HF model)
- **Class imbalance:** weighted cross-entropy with exponent-scaled weights
  (`--weight_exp`, default 0.7), macro-F1 as the model-selection metric,
  early stopping (`--patience`, default 5)
- **Tracking:** parameters, per-epoch validation metrics, test metrics and per-class F1
  logged to MLflow (experiment `scidcc_classification`); the best model is registered
  as `SciDCC-Classifier` and promoted to the `production` alias only if its test
  macro-F1 beats the current production version

```bash
python -m training.classification.train \
  --input body --strategy full_ft --lr 2e-5 --batch 16 --epochs 30
```

Prerequisites: the preprocessed HF datasets and `label_map.json` in `data/processed/`
(produced by the notebooks in `notebooks/`).

To point the backend at a newly trained checkpoint, set `CLS_MODEL_PATH` in `.env`
and restart the backend.

---

## Hardware requirements (GLiNER fine-tuning)

| Setup | Training time (10 epochs, ~1500 samples) |
|---|---|
| RTX 3050 8 GB, batch=8 | ~5–10 minutes |
| CPU only | ~30–60 minutes |

If you run out of GPU memory, reduce `--batch` to `4`.

---

## Tests

The span-conversion logic (the most bug-prone part of the pipeline) has a pytest
regression suite, including a check over the full real annotation file:

```bash
python -m pytest tests/ -v
```
