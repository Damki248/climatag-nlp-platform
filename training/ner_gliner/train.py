#!/usr/bin/env python3
"""
GLiNER fine-tuning – adds the Climate Model category on top of the GLiNER baseline.
Uses experience replay: CliReNERsilver dataset (existing categories) + Climate Model category annotations.
Ratio: ~5:1 following the recommentation in GLiNER GitHub issue #163.
"""
 
import argparse
import json
import logging
import os
import random
from pathlib import Path
 
import pandas as pd
import torch
import mlflow
from dotenv import load_dotenv
from gliner import GLiNER
from gliner.training import TrainingArguments
 
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
 
ALL_LABELS = [
    "Asset", "Body Part", "Body of Water", "Chemical", "Disease",
    "Ecosystem", "Energy Source", "Field of Study", "Geographical Feature",
    "Intellectual Artefact", "Location", "Mathematical Expression",
    "Measuring Device", "Meteorological Phenomenon", "Method",
    "Natural Disaster", "Natural Phenomenon", "Organism", "Organization",
    "Other", "Person", "Physical Artefact", "Physical Phenomenon",
    "Policy", "Quantity", "Satellite", "System", "Time Period",
    "Climate Model",
]
 
# IOB2 index -> label mapping from SILVER dataset
IDX2LABEL = [
    "O", "B-Asset", "I-Asset", "B-Body Part", "I-Body Part",
    "B-Body of Water", "I-Body of Water", "B-Chemical", "I-Chemical",
    "B-Disease", "I-Disease", "B-Ecosystem", "I-Ecosystem",
    "B-Energy Source", "I-Energy Source", "B-Field of Study", "I-Field of Study",
    "B-Geographical Feature", "I-Geographical Feature", "B-Intellectual Artefact",
    "I-Intellectual Artefact", "B-Location", "I-Location",
    "B-Mathematical Expression", "I-Mathematical Expression",
    "B-Measuring Device", "I-Measuring Device", "B-Meteorological Phenomenon",
    "I-Meteorological Phenomenon", "B-Method", "I-Method",
    "B-Natural Disaster", "I-Natural Disaster", "B-Natural Phenomenon",
    "I-Natural Phenomenon", "B-Organism", "I-Organism",
    "B-Organization", "I-Organization", "B-Other", "I-Other",
    "B-Person", "I-Person", "B-Physical Artefact", "I-Physical Artefact",
    "B-Physical Phenomenon", "I-Physical Phenomenon", "B-Policy", "I-Policy",
    "B-Quantity", "I-Quantity", "B-Satellite", "I-Satellite",
    "B-System", "I-System", "B-Time Period", "I-Time Period",
]
 
 
def parse_args():
    p = argparse.ArgumentParser(description="GLiNER fine-tuning – Climate Model category")
    p.add_argument("--silver_train",        default="data/raw/CliReNER_SILVER/train-00000-of-00001.parquet")
    p.add_argument("--cm_annotations",      default="data/annotations/climate_model_annotations.json")
    p.add_argument("--base_model",          default="models/ner_gliner_baseline")
    p.add_argument("--output_model",        default="models/ner_gliner_climate_model")
    p.add_argument("--epochs",              default=10,   type=int)
    p.add_argument("--lr",                  default=5e-6, type=float)
    p.add_argument("--batch",               default=8,    type=int)
    p.add_argument("--silver_ratio",        default=5,    type=int,
                   help="How many times more CliReNERsilver samples than Climate Model samples")
    p.add_argument("--val_split",           default=0.15, type=float)
    p.add_argument("--seed",                default=42,   type=int)
    p.add_argument("--experiment",          default="climtag_ner_gliner")
    p.add_argument("--run_name",            default="gliner_climate_model_v1")
    return p.parse_args()
 
 
def iob2_to_gliner(tokens: list, ner_tags: list) -> dict:
    """Converts IOB2 format (SILVER) into GLiNER format."""
    entities = []
    i = 0
    while i < len(ner_tags):
        tag = IDX2LABEL[ner_tags[i]]
        if tag.startswith("B-"):
            label = tag[2:]
            start = i
            end = i + 1
            while end < len(ner_tags) and IDX2LABEL[ner_tags[end]] == f"I-{label}":
                end += 1
            entities.append((start, end, label))
            i = end
        else:
            i += 1
    return {"tokenized_text": tokens, "ner": entities}
 
 
def parse_climate_model_annotations(path: str) -> list:
    """
    Parses Label Studio export with Climate Model annotations.
    Converts char spans -> token spans into GLiNER format.
    """
    with open(path) as f:
        data = json.load(f)
 
    samples = []
    skipped = 0
    for task in data:
        text = task["data"]["text"]
        annotations = task.get("annotations", [])
        if not annotations:
            skipped += 1
            continue
 
        result = annotations[0].get("result", [])
        if not result:
            skipped += 1
            continue
 
        # Whitespace tokenization with position
        words = text.split()
        token_starts = []
        token_ends = []
        pos = 0
        for word in words:
            idx = text.find(word, pos)
            token_starts.append(idx)
            token_ends.append(idx + len(word))
            pos = idx + len(word)
 
        # Convert char spans -> token spans
        entities = []
        for r in result:
            if not r.get("value", {}).get("labels"):
                continue
            e_start = r["value"]["start"]
            e_end   = r["value"]["end"]
            label   = r["value"]["labels"][0]
 
            tok_start = None
            tok_end   = None
            for j, (ts, te) in enumerate(zip(token_starts, token_ends)):
                if ts >= e_start and tok_start is None:
                    tok_start = j
                if te <= e_end:
                    tok_end = j + 1
 
            if tok_start is not None and tok_end is not None:
                entities.append((tok_start, tok_end, label))
 
        if entities:
            samples.append({"tokenized_text": words, "ner": entities})
        else:
            skipped += 1
 
    log.info("Parsed %d Climate Model samples, skipped %d", len(samples), skipped)
    return samples
 
 
def tokens_to_char_spans(sample: dict) -> set:
    """
    Converts token-level span indexes from sample["ner"] into character offsets.
    GLiNER predict_entities returns character offsets, so format should be aligned.
    """
    words = sample["tokenized_text"]
    text  = " ".join(words)
 
    # Build mapping: token_i -> char_start, char_end
    char_starts, char_ends = [], []
    pos = 0
    for word in words:
        idx = text.find(word, pos)
        char_starts.append(idx)
        char_ends.append(idx + len(word))
        pos = idx + len(word)
 
    spans = set()
    for (tok_s, tok_e, lbl) in sample["ner"]:
        if lbl == "Climate Model":
            spans.add((char_starts[tok_s], char_ends[tok_e - 1]))
    return spans
 
 
def evaluate_climate_model(model: GLiNER, cm_samples: list, n: int = 50) -> dict:
    """
    Evaluation on Climate Model samples (held-out test set).
 
    Uses overlap-based matching because annotators aren't consistent in span
    borders (for example, 'CMIP6' vs 'CMIP6 models'). 
    Prediction is calculated as TP if it overlaps with ground truth span and has the right label.
    """
    tp = fp = fn = 0
    eval_samples = cm_samples[:n]
 
    for sample in eval_samples:
        text  = " ".join(sample["tokenized_text"])
        preds = model.predict_entities(text, ALL_LABELS, threshold=0.5)
 
        # Character offsets from GLiNER
        pred_spans = [
            (e["start"], e["end"]) for e in preds if e["label"] == "Climate Model"
        ]
        # Token indices → character offsets
        true_spans = list(tokens_to_char_spans(sample))
 
        # Overlap matching: a prediction is TP if it overlaps any true span
        matched_true = set()
        matched_pred = set()
        for i, p in enumerate(pred_spans):
            for j, t in enumerate(true_spans):
                if p[0] < t[1] and t[0] < p[1]:   # overlap condition
                    matched_true.add(j)
                    matched_pred.add(i)
                    break  # each prediction matches at most one true span
        tp += len(matched_pred)
        fp += len(pred_spans) - len(matched_pred)
        fn += len(true_spans) - len(matched_true)
 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
 
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
 
 
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
 
    # 1. Load SILVER dataset
    log.info("Loading SILVER dataset: %s", args.silver_train)
    df = pd.read_parquet(args.silver_train)
    silver_samples = [
        iob2_to_gliner(list(row["tokens"]), list(row["ner_tags"]))
        for _, row in df.iterrows()
    ]
    log.info("SILVER samples: %d", len(silver_samples))
 
    # 2. Load Climate Model annotations
    log.info("Loading Climate Model annotations: %s", args.cm_annotations)
    cm_samples = parse_climate_model_annotations(args.cm_annotations)
 
    if not cm_samples:
        log.error("No Climate Model samples. Check %s", args.cm_annotations)
        return
 
    # 3. CM held-out test split (20%) — these samples never enter training
    random.shuffle(cm_samples)
    n_cm_test  = max(10, int(len(cm_samples) * 0.20))
    cm_test    = cm_samples[:n_cm_test]
    cm_train   = cm_samples[n_cm_test:]
    log.info("Climate Model: train: %d, held-out test: %d", len(cm_train), len(cm_test))
 
    # 4. Experience replay mix (only cm_train, not cm_test)
    n_silver = min(args.silver_ratio * len(cm_train), len(silver_samples))
    random.shuffle(silver_samples)
    mixed = silver_samples[:n_silver] + cm_train
    random.shuffle(mixed)
    log.info("Total samples: %d (CliReNERsilver: %d, Climate Model train: %d)",
             len(mixed), n_silver, len(cm_train))
 
    # 5. Train/val split on mixed set
    n_val      = max(1, int(len(mixed) * args.val_split))
    val_data   = mixed[:n_val]
    train_data = mixed[n_val:]
    log.info("Train: %d, Val: %d", len(train_data), len(val_data))
 
    # 5. Load the model
    log.info("Loading GLiNER model: %s", args.base_model)
    model = GLiNER.from_pretrained(args.base_model)
 
    # 6. MLflow setup
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    try:
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(args.experiment)
        if exp is None:
            client.create_experiment(
                args.experiment,
                artifact_location=f"mlflow-artifacts:/{args.experiment}",
            )
    except Exception as e:
        log.warning("MLflow experiment setup: %s", e)
    mlflow.set_experiment(args.experiment)
 
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "base_model":    args.base_model,
            "epochs":        args.epochs,
            "lr":            args.lr,
            "batch":         args.batch,
            "silver_ratio":  args.silver_ratio,
            "train_size":    len(train_data),
            "val_size":      len(val_data),
            "cm_train":      len(cm_train),
            "cm_test":       len(cm_test),
            "seed":          args.seed,
        })
 
        # 7. Fine-tuning
        log.info("Starting fine-tuning (%d epochs)...", args.epochs)
        training_args = TrainingArguments(
            output_dir=args.output_model + "_checkpoints",
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            warmup_steps=100,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            report_to="none",
        )
 
        model.train_model(
            train_dataset=train_data,
            eval_dataset=val_data,
            training_args=training_args,
            output_dir=args.output_model + "_checkpoints",
        )
 
        # 8. Save the final model
        output_path = Path(args.output_model)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path))
        log.info("Model saved: %s", output_path)
 
        # 9. Evaluate Climate Model category, only on held-out test set
        log.info("Evaluating Climate Model category on %d held-out samples...", len(cm_test))
        eval_metrics = evaluate_climate_model(model, cm_test, n=len(cm_test))
        log.info("Climate Model (held-out) – Precision: %.4f, Recall: %.4f, F1: %.4f",
                 eval_metrics["precision"], eval_metrics["recall"], eval_metrics["f1"])
 
        mlflow.log_metrics({
            "cm_precision": eval_metrics["precision"],
            "cm_recall":    eval_metrics["recall"],
            "cm_f1":        eval_metrics["f1"],
        })
 
    log.info("Fine-tuning complete.")
 
 
if __name__ == "__main__":
    main()