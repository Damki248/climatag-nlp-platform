#!/usr/bin/env python3
"""
SpanMarker fine-tuning s SILVER datasetom + Climate Model anotacije.
Strategija: merge SILVER (28 klasa) + Climate Model rečenice (29. klasa)
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import pandas as pd
import mlflow
import torch
from dotenv import load_dotenv
from span_marker import SpanMarkerModel, Trainer
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ALL_LABELS = [
    "O",
    "Asset", "Body Part", "Body of Water", "Chemical", "Disease",
    "Ecosystem", "Energy Source", "Field of Study", "Geographical Feature",
    "Intellectual Artefact", "Location", "Mathematical Expression",
    "Measuring Device", "Meteorological Phenomenon", "Method",
    "Natural Disaster", "Natural Phenomenon", "Organism", "Organization",
    "Other", "Person", "Physical Artefact", "Physical Phenomenon",
    "Policy", "Quantity", "Satellite", "System", "Time Period",
    "Climate Model"
]

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
    "B-System", "I-System", "B-Time Period", "I-Time Period"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--silver_train",      default="data/raw/CliReNER_SILVER/train-00000-of-00001.parquet")
    p.add_argument("--silver_val",        default="data/raw/CliReNER_SILVER/validation-00000-of-00001.parquet")
    p.add_argument("--cm_annotations",   default="data/annotations/climate_model_annotations.json")
    p.add_argument("--base_model",        default="P0L3/cliscibert_scivocab_uncased")
    p.add_argument("--output_model",      default="models/ner_adapted_silver")
    p.add_argument("--epochs",            default=20,   type=int)
    p.add_argument("--lr",                default=5e-5, type=float)
    p.add_argument("--batch",             default=8,    type=int)
    p.add_argument("--seed",              default=42,   type=int)
    p.add_argument("--experiment",        default="clirener_silver_cm")
    p.add_argument("--run_name",          default="spanmarker_silver_cm_v1")
    return p.parse_args()


def silver_to_spanmarker(tokens: list, ner_tags: list) -> dict:
    """Konvertira SILVER IOB2 format u SpanMarker format."""
    iob2_tags = [IDX2LABEL[int(t)] for t in ner_tags]
    return {
        "tokens": [str(t) for t in tokens],
        "ner_tags": iob2_tags
    }


def parse_cm_annotations(path: str) -> list:
    """Parsira Label Studio export u SpanMarker format."""
    with open(path) as f:
        data = json.load(f)

    samples = []
    for task in data:
        text = task["data"]["text"]
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        result = annotations[0].get("result", [])
        if not result:
            continue

        words = text.split()
        token_starts = []
        token_ends = []
        pos = 0
        for word in words:
            idx = text.find(word, pos)
            token_starts.append(idx)
            token_ends.append(idx + len(word))
            pos = idx + len(word)

        ner_tags = ["O"] * len(words)

        for r in result:
            if not r.get("value", {}).get("labels"):
                continue
            e_start = r["value"]["start"]
            e_end = r["value"]["end"]
            label = r["value"]["labels"][0]

            first = True
            for j, (ts, te) in enumerate(zip(token_starts, token_ends)):
                if ts >= e_start and te <= e_end:
                    ner_tags[j] = f"B-{label}" if first else f"I-{label}"
                    first = False

        if any(t != "O" for t in ner_tags):
            samples.append({"tokens": words, "ner_tags": ner_tags})

    log.info("Parsirano %d Climate Model uzoraka", len(samples))
    return samples


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Učitaj SILVER train
    log.info("Učitavam SILVER train dataset...")
    df_train = pd.read_parquet(args.silver_train)
    df_val = pd.read_parquet(args.silver_val)

    silver_train = [silver_to_spanmarker(list(r["tokens"]), list(r["ner_tags"])) for _, r in df_train.iterrows()]
    silver_val = [silver_to_spanmarker(list(r["tokens"]), list(r["ner_tags"])) for _, r in df_val.iterrows()]
    log.info("SILVER train: %d, val: %d", len(silver_train), len(silver_val))

    # 2. Učitaj Climate Model anotacije
    log.info("Učitavam Climate Model anotacije...")
    cm_samples = parse_cm_annotations(args.cm_annotations)

    # 3. Split CM uzoraka na train/val (85/15)
    random.shuffle(cm_samples)
    n_val = max(1, int(len(cm_samples) * 0.15))
    cm_val = cm_samples[:n_val]
    cm_train = cm_samples[n_val:]
    log.info("CM train: %d, val: %d", len(cm_train), len(cm_val))

    # 4. Merge
    train_data = silver_train + cm_train
    val_data = silver_val + cm_val
    random.shuffle(train_data)
    log.info("Ukupno train: %d, val: %d", len(train_data), len(val_data))

    # 5. Dataset
    ds = DatasetDict({
        "train": Dataset.from_list(train_data),
        "eval":  Dataset.from_list(val_data),
    })

    # 6. Učitaj model s overwrite_entities
    log.info("Učitavam baseline model...")
    model = SpanMarkerModel.from_pretrained(
        args.base_model,
        labels=ALL_LABELS,
        model_max_length=256,
        entity_max_length=8,
        overwrite_entities=True,
    )

    # 7. MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "base_model": args.base_model,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch": args.batch,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "seed": args.seed,
        })

        training_args = TrainingArguments(
            output_dir=args.output_model + "_checkpoints",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_overall_f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            seed=args.seed,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["eval"],
        )

        log.info("Pokrećem fine-tuning...")
        trainer.train()

        metrics = trainer.evaluate()
        log.info("Eval metrics: %s", metrics)
        mlflow.log_metrics({
            k.replace("eval_", ""): v
            for k, v in metrics.items()
            if isinstance(v, float)
        })

        output_path = Path(args.output_model)
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_path))
        log.info("Model spremljen: %s", output_path)

    log.info("Fine-tuning završen.")


if __name__ == "__main__":
    main()
