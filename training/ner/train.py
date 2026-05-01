#!/usr/bin/env python3
"""
NER fine-tuning script – dodaje novu SDG kategoriju na baseline CliReNER model.

Usage:
    python training/ner/train.py \
        --annotations data/annotations/sdg_annotations.json \
        --base_model models/ner_baseline \
        --output_model models/ner_adapted \
        --epochs 5 \
        --lr 1e-5

Strategija:
    - Parsamo Label Studio export, izvlačimo samo SDG anotacije (origin=manual)
    - Konvertiramo u SpanMarker format (lista diktova s tokens + ner_tags)
    - Fine-tunamo baseline model s novom SDG kategorijom
    - Čuvamo novi model u output_model direktoriju
"""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import torch
from dotenv import load_dotenv
from span_marker import SpanMarkerModel, Trainer
from datasets import Dataset, DatasetDict

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="NER fine-tuning – dodavanje SDG kategorije")
    p.add_argument("--annotations",   default="data/annotations/sdg_annotations.json")
    p.add_argument("--base_model",    default="models/ner_baseline")
    p.add_argument("--output_model",  default="models/ner_adapted")
    p.add_argument("--epochs",        default=5,    type=int)
    p.add_argument("--lr",            default=1e-5, type=float)
    p.add_argument("--batch",         default=4,    type=int)
    p.add_argument("--val_split",     default=0.15, type=float,
                   help="Udio SDG podataka za validaciju")
    p.add_argument("--no_eval",       action="store_true",
                   help="Treniraj bez evaluacije, sve uzorke stavi u train")
    p.add_argument("--freeze_encoder",  action="store_true",
                   help="Zamrzni encoder, treniraj samo klasifikacijsku glavu")
    p.add_argument("--experiment",    default="clirener_sdg")
    p.add_argument("--run_name",      default="ner_sdg_finetune")
    p.add_argument("--seed",          default=42,   type=int)
    return p.parse_args()


# ------------------------------------------------------------------ #
# Parsanje Label Studio exporta
# ------------------------------------------------------------------ #

def parse_label_studio(annotations_path: str) -> list[dict]:
    """
    Izvlači SDG anotacije iz Label Studio JSON exporta.
    Vraća listu diktova: {text, entities: [{start, end, label}]}
    Filtrira samo taskove koji imaju SDG anotacije (origin=manual).
    """
    with open(annotations_path) as f:
        data = json.load(f)

    samples = []
    skipped = 0

    for task in data:
        text = task["data"]["text"]
        annotations = task.get("annotations", [])
        if not annotations:
            skipped += 1
            continue

        # uzmi prvu anotaciju
        result = annotations[0].get("result", [])

        # filtriraj samo SDG entitete (manual origin)
        sdg_entities = [
            {
                "start": r["value"]["start"],
                "end":   r["value"]["end"],
                "label": r["value"]["labels"][0],
            }
            for r in result
            if r.get("origin") == "manual"
            and r.get("value", {}).get("labels")
            and r["value"]["labels"][0] == "SDG"
        ]

        if not sdg_entities:
            skipped += 1
            continue

        samples.append({"text": text, "entities": sdg_entities})

    log.info("Parsirano %d SDG uzoraka, preskočeno %d taskova bez SDG anotacija", len(samples), skipped)
    return samples


# ------------------------------------------------------------------ #
# Konverzija u SpanMarker format
# ------------------------------------------------------------------ #

def to_span_marker_format(samples: list[dict]) -> list[dict]:
    """
    Konvertira entity span format u SpanMarker token+tag format.
    SpanMarker prima: {"tokens": [...], "ner_tags": [...]}
    gdje su ner_tags IOB2 labele ili span labele.

    SpanMarker 1.5+ prima direktno character spans kao:
    {"document_id": ..., "sentence_id": ..., "tokens": [...], "ner_tags": [...]}

    Koristimo jednostavni whitespace tokenizer i mapiramo character spans na tokene.
    """
    converted = []

    for i, sample in enumerate(samples):
        text = sample["text"]
        entities = sample["entities"]

        # whitespace tokenizacija s pozicijama
        tokens = []
        token_starts = []
        token_ends = []
        current = 0
        for word in text.split():
            start = text.index(word, current)
            end = start + len(word)
            tokens.append(word)
            token_starts.append(start)
            token_ends.append(end)
            current = end

        # IOB2 tagiranje
        ner_tags = ["O"] * len(tokens)

        for entity in entities:
            e_start = entity["start"]
            e_end = entity["end"]
            label = entity["label"]

            first = True
            for j, (t_start, t_end) in enumerate(zip(token_starts, token_ends)):
                # token se preklapa s entitetom
                if t_start >= e_start and t_end <= e_end:
                    if first:
                        ner_tags[j] = f"B-{label}"
                        first = False
                    else:
                        ner_tags[j] = f"I-{label}"
                elif t_start < e_end and t_end > e_start:
                    # parcijalno preklapanje – označi kao B ako je početak
                    if first:
                        ner_tags[j] = f"B-{label}"
                        first = False

        converted.append({
            "tokens":   tokens,
            "ner_tags": ner_tags,
        })

    return converted


# ------------------------------------------------------------------ #
# Train/val split
# ------------------------------------------------------------------ #

def split_dataset(samples: list[dict], val_split: float, seed: int, no_eval: bool = False) -> DatasetDict:
    import random
    random.seed(seed)
    random.shuffle(samples)
    if no_eval or val_split <= 0:
        log.info("Train: %d, Val: 0 (no_eval mode)", len(samples))
        return DatasetDict({"train": Dataset.from_list(samples)})
    n_val = max(1, int(len(samples) * val_split))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    log.info("Train: %d, Val: %d", len(train_samples), len(val_samples))
    return DatasetDict({
        "train": Dataset.from_list(train_samples),
        "eval":  Dataset.from_list(val_samples),
    })


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 1. Parsiraj anotacije
    log.info("Parsiranje Label Studio exporta: %s", args.annotations)
    samples = parse_label_studio(args.annotations)
    if not samples:
        log.error("Nema SDG uzoraka u exportu. Provjeri annotations file.")
        return

    # 2. Konvertiraj u SpanMarker format
    converted = to_span_marker_format(samples)

    # 3. Napravi dataset
    ds = split_dataset(converted, args.val_split, args.seed, no_eval=args.no_eval)

    # 4. Učitaj baseline model
    log.info("Učitavam baseline model: %s", args.base_model)

    # postojeće labele baseline modela + nova SDG kategorija
    existing_labels = [
        "O", "Asset", "Body Part", "Body of Water", "Chemical", "Disease",
        "Ecosystem", "Energy Source", "Field of Study", "Geographical Feature",
        "Intellectual Artefact", "Location", "Mathematical Expression",
        "Measuring Device", "Meteorological Phenomenon", "Method",
        "Natural Disaster", "Natural Phenomenon", "Organism", "Organization",
        "Other", "Person", "Physical Artefact", "Physical Phenomenon",
        "Policy", "Quantity", "Satellite", "System", "Time Period",
        "SDG",  # nova kategorija
    ]

    model = SpanMarkerModel.from_pretrained(
        args.base_model,
        labels=existing_labels,
        model_max_length=256,
        entity_max_length=8,
        overwrite_entities=True,
    )

    # 4b. Zamrzni encoder ako je zadano
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info("Encoder zamrznut. Trainable: %d / %d params", trainable, total)

    # 5. MLflow tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "base_model":  args.base_model,
            "epochs":      args.epochs,
            "lr":          args.lr,
            "batch":       args.batch,
            "train_size":  len(ds["train"]),
            "val_size":    len(ds["eval"]) if "eval" in ds else 0,
            "seed":        args.seed,
            "freeze_encoder": args.freeze_encoder,
        })

        # 6. Trainer
        from transformers import TrainingArguments

        no_eval = args.no_eval or "eval" not in ds

        training_args = TrainingArguments(
            output_dir=args.output_model + "_checkpoints",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            eval_strategy="no" if no_eval else "epoch",
            save_strategy="epoch",
            load_best_model_at_end=False if no_eval else True,
            metric_for_best_model=None if no_eval else "eval_overall_f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            seed=args.seed,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds.get("eval", None),
        )

        log.info("Pokrećem fine-tuning...")
        trainer.train()

        # 7. Evaluacija (samo ako ima eval set)
        if not no_eval:
            metrics = trainer.evaluate()
            log.info("Eval metrics: %s", metrics)
            mlflow.log_metrics({
                k.replace("eval_", ""): v
                for k, v in metrics.items()
                if isinstance(v, float)
            })
        else:
            log.info("no_eval mode – preskačem evaluaciju.")

        # 8. Spremi model
        output_path = Path(args.output_model)
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_path))
        log.info("Adapted model spremljen: %s", output_path)

        mlflow.log_param("output_model", str(output_path))

    log.info("Fine-tuning završen.")


if __name__ == "__main__":
    main()