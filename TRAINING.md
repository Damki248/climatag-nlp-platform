# Training Guide

This guide covers running classification fine-tuning experiments on the SciDCC dataset.

> **NER model training** is not covered here – the NER model (CliReNER) is used as a pre-trained baseline without further fine-tuning in this project.

---

## Prerequisites

- Conda environment `spanmarker-env` activated
- SciDCC dataset preprocessed (see [SETUP.md](SETUP.md) step 5)
- MLflow running (`docker compose -f docker/docker-compose.yml up -d`)

---

## Classification training script

All experiments run through a single script:

```bash
conda activate spanmarker-env
cd ~/climate-nlp-platform

python training/classification/train.py [OPTIONS]
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | `title_summary` | Input text field: `body` or `title_summary` |
| `--strategy` | `full_ft` | Fine-tuning strategy: `full_ft` or `lora` |
| `--model` | `P0L3/SciClimateBERT` | HuggingFace model ID or local path |
| `--lr` | `5e-6` | Learning rate |
| `--epochs` | `30` | Maximum training epochs |
| `--batch` | `8` | Batch size per device |
| `--warmup` | `0.2` | Warmup ratio (fraction of total steps) |
| `--patience` | `5` | Early stopping patience (epochs) |
| `--weight_exp` | `0.7` | Class weight exponent (0.7 = moderate, 1.0 = fully balanced) |
| `--experiment` | `scidcc_classification` | MLflow experiment name |
| `--run_name` | auto | MLflow run name |

### LoRA-specific arguments

| Argument | Default | Description |
|---|---|---|
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha scaling |
| `--lora_dropout` | `0.1` | LoRA dropout |

---

## Running experiments

### Best known configuration (full fine-tuning, body input)

This is the current best run with macro F1 = **0.4208** on the test set:

```bash
python training/classification/train.py \
  --input body \
  --strategy full_ft \
  --lr 2e-5 \
  --epochs 30 \
  --batch 16 \
  --warmup 0.15 \
  --patience 10 \
  --weight_exp 0.7 \
  --experiment scidcc_cls \
  --run_name full_ft_body_lr2e5_best
```

The best checkpoint is saved to `models/cls_full_ft_best/` automatically.

### LoRA experiment

```bash
python training/classification/train.py \
  --input body \
  --strategy lora \
  --lr 3e-4 \
  --epochs 30 \
  --batch 16 \
  --warmup 0.1 \
  --patience 10 \
  --weight_exp 0.7 \
  --lora_r 16 \
  --lora_alpha 32 \
  --experiment scidcc_cls \
  --run_name lora_body_r16_lr3e4
```

### Title+Summary input experiment

```bash
python training/classification/train.py \
  --input title_summary \
  --strategy full_ft \
  --lr 2e-5 \
  --epochs 30 \
  --batch 16 \
  --warmup 0.15 \
  --patience 10 \
  --weight_exp 0.7 \
  --experiment scidcc_cls \
  --run_name full_ft_title_summary_lr2e5
```

---

## Tracked metrics (MLflow)

Every run logs the following to MLflow at http://localhost:5000:

**Parameters (logged once):**
- model, input, strategy, lr, epochs, batch, warmup, patience, weight_exp, seed, trainable_params

**Per-epoch validation metrics:**
- `val_macro_f1`, `val_weighted_f1`, `val_accuracy`, `val_loss`

**Test metrics (end of run):**
- `test_macro_f1`, `test_weighted_f1`, `test_accuracy`
- `test_f1_{category}` for each of the 20 SciDCC categories

---

## Results summary

| Run | Input | Strategy | LR | Test macro F1 |
|---|---|---|---|---|
| exp07 (best) | body | full_ft | 2e-5 | **0.4208** |
| patience10 | body | full_ft | 2e-5 | 0.3865 |

> **Baseline from paper (Spokoyny et al.):** macro F1 = 53.75% (different model/setup)

---

## Hardware requirements

| Setup | Training time (est.) |
|---|---|
| RTX 3050 8GB, batch=16 | ~3–5 min/epoch |
| CPU only | Very slow – not recommended for full runs |

If you run out of GPU memory, reduce `--batch` to `8` or `4`.

---

## Output files

After a training run completes, you'll find:

```
models/
└── cls_full_ft_best/       # Best checkpoint (by val macro F1)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.json

training/classification/
└── results/
    ├── per_class_f1.json   # Per-category F1 on test set
    └── test_results.json   # Full test metrics
```

The backend (`cls_service.py`) always loads from `models/cls_full_ft_best/` on startup.
