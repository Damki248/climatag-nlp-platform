# training/classification/train.py

import argparse
import json
import logging
import os
from pathlib import Path
import re

import evaluate
import mlflow
import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="SciDCC classification fine-tuning")
    p.add_argument("--input",       default="title_summary", choices=["title_summary", "body"],
                   help="Koji HF dataset koristimo (hf_title_summary ili hf_body)")
    p.add_argument("--strategy",    default="full_ft",       choices=["full_ft", "lora"],
                   help="Strategija fine-tuninga")
    p.add_argument("--model",       default="P0L3/SciClimateBERT",
                   help="HuggingFace model ID ili lokalni path")
    p.add_argument("--lr",          default=5e-6,  type=float)
    p.add_argument("--epochs",      default=30,    type=int)
    p.add_argument("--batch",       default=8,     type=int)
    p.add_argument("--warmup",      default=0.2,   type=float)
    p.add_argument("--patience",    default=5,     type=int,
                   help="Early stopping patience (epohe)")
    p.add_argument("--weight_exp",  default=0.7,   type=float,
                   help="Eksponent za class weight skaliranje (0.7=agresivno, 1.0=balanced)")
    p.add_argument("--lora_r",      default=16,    type=int)
    p.add_argument("--lora_alpha",  default=32,    type=int)
    p.add_argument("--lora_dropout",default=0.1,   type=float)
    p.add_argument("--experiment",  default="scidcc_classification",
                   help="MLflow experiment name")
    p.add_argument("--run_name",    default=None,
                   help="MLflow run name (auto ako nije zadano)")
    p.add_argument("--seed",        default=42,    type=int)
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--models_dir",    default="models")
    return p.parse_args()


# ------------------------------------------------------------------ #
# Class weights
# ------------------------------------------------------------------ #

def compute_weights(train_dataset, num_labels, exponent):
    labels = np.array(train_dataset["label"])
    counts = np.array([int((labels == i).sum()) for i in range(num_labels)])
    counts = np.where(counts == 0, 1, counts)  # izbjegni dijeljenje s nulom
    weights = 1.0 / np.power(counts, exponent)
    weights = weights / weights.mean()
    log.info("Class weights (min=%.4f, max=%.4f)", weights.min(), weights.max())
    return torch.tensor(weights, dtype=torch.float32)


# ------------------------------------------------------------------ #
# Metrike
# ------------------------------------------------------------------ #

metric_f1  = evaluate.load("f1")
metric_acc = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "macro_f1":    round(metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"], 4),
        "weighted_f1": round(metric_f1.compute(predictions=preds, references=labels, average="weighted")["f1"], 4),
        "accuracy":    round(metric_acc.compute(predictions=preds, references=labels)["accuracy"], 4),
    }


# ------------------------------------------------------------------ #
# Custom Trainer s weighted lossom
# ------------------------------------------------------------------ #

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    PROCESSED_DIR = Path(args.processed_dir)
    MODELS_DIR    = Path(args.models_dir)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # label mapping
    with open(PROCESSED_DIR / "label_map.json") as f:
        label_map = json.load(f)
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    num_labels = len(label2id)

    # dataset
    ds_key = f"hf_{args.input}"
    ds = load_from_disk(str(PROCESSED_DIR / ds_key))
    log.info("Dataset: %s  train=%d val=%d test=%d", ds_key, len(ds["train"]), len(ds["val"]), len(ds["test"]))

    # class weights
    class_weights = compute_weights(ds["train"], num_labels, args.weight_exp)

    # tokenizer i collator
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator  = DataCollatorWithPadding(tokenizer=tokenizer)

    # output dir za ovaj run
    run_label  = args.run_name or f"{args.strategy}_{args.input}_lr{args.lr}_e{args.weight_exp}"
    output_dir = MODELS_DIR / f"cls_{run_label}"

    # MLflow setup
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Kreiraj eksperiment s HTTP artifact location (izbjegava permission error)
    client_setup = mlflow.MlflowClient()
    try:
        exp = client_setup.get_experiment_by_name(args.experiment)
        if exp is None:
            client_setup.create_experiment(
                args.experiment,
                artifact_location=f"mlflow-artifacts:/{args.experiment}"
            )
    except Exception:
        pass
    mlflow.set_experiment(args.experiment)
    
    with mlflow.start_run(run_name=run_label):

        # logiraj sve argumente kao params
        mlflow.log_params({
            "model":        args.model,
            "input":        args.input,
            "strategy":     args.strategy,
            "lr":           args.lr,
            "epochs":       args.epochs,
            "batch":        args.batch,
            "warmup":       args.warmup,
            "patience":     args.patience,
            "weight_exp":   args.weight_exp,
            "seed":         args.seed,
            **({"lora_r": args.lora_r, "lora_alpha": args.lora_alpha} if args.strategy == "lora" else {}),
        })

        # model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        if args.strategy == "lora":
            lora_cfg = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["query", "value"],
                bias="none",
            )
            model = get_peft_model(base_model, lora_cfg)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            log.info("LoRA trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)
            mlflow.log_param("trainable_params", trainable)
        else:
            model = base_model
            mlflow.log_param("trainable_params", sum(p.numel() for p in model.parameters()))

        # training args
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=32,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=args.warmup,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            seed=args.seed,
            logging_steps=50,
            report_to="none",  # rucno logiramo u MLflow ispod
            save_total_limit=2,
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["val"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )

        # treniranje
        log.info("Pocinjem %s treniranje...", args.strategy)
        trainer.train()

        # logiraj per-epoch metrike u MLflow
        for entry in trainer.state.log_history:
            if "eval_macro_f1" in entry:
                step = int(entry["epoch"])
                mlflow.log_metrics({
                    "val_macro_f1":    entry["eval_macro_f1"],
                    "val_weighted_f1": entry["eval_weighted_f1"],
                    "val_accuracy":    entry["eval_accuracy"],
                    "val_loss":        entry["eval_loss"],
                }, step=step)

        # evaluacija na test setu
        log.info("Evaluacija na test setu...")
        test_results = trainer.evaluate(ds["test"])
        preds_out    = trainer.predict(ds["test"])
        pred_labels  = np.argmax(preds_out.predictions, axis=1)
        true_labels  = preds_out.label_ids

        per_class_f1 = metric_f1.compute(
            predictions=pred_labels,
            references=true_labels,
            average=None,
        )["f1"]

        # logiraj test metrike
        mlflow.log_metrics({
            "test_macro_f1":    test_results["eval_macro_f1"],
            "test_weighted_f1": test_results["eval_weighted_f1"],
            "test_accuracy":    test_results["eval_accuracy"],
        })

        # logiraj per-class F1 kao metriku i kao artifact
        per_class_dict = {id2label[i]: round(float(f), 4) for i, f in enumerate(per_class_f1)}
        for cat, f1 in per_class_dict.items():
            safe_cat = re.sub(r'[^a-zA-Z0-9_\-\. :/]', '', cat).replace(' ', '_')
            mlflow.log_metric(f"test_f1_{safe_cat}", f1)

        # spremi per-class JSON kao artifact
        pc_path = output_dir / "per_class_f1.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(pc_path, "w") as f:
            json.dump(per_class_dict, f, indent=2)

        # spremi model
        best_path = MODELS_DIR / f"cls_{run_label}_best"
        trainer.save_model(str(best_path))
        tokenizer.save_pretrained(str(best_path))

        # spremi test_results.json
        result_summary = {
            "test_macro_f1":    test_results["eval_macro_f1"],
            "test_weighted_f1": test_results["eval_weighted_f1"],
            "test_accuracy":    test_results["eval_accuracy"],
            "input":            args.input,
            "model":            args.model,
            "strategy":         args.strategy,
            "lr":               args.lr,
            "weight_exp":       args.weight_exp,
        }
        with open(best_path / "test_results.json", "w") as f:
            json.dump(result_summary, f, indent=2)

        log.info("Test macro F1: %.4f", test_results["eval_macro_f1"])
        log.info("Model spreman: %s", best_path)

        test_macro_f1 = test_results["eval_macro_f1"]
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/cls_model"
        
        mlflow.transformers.log_model(
            transformers_model={
                "model": trainer.model,
                "tokenizer": tokenizer,
            },
            artifact_path="cls_model",
            task="text-classification",
        )

        registered = mlflow.register_model(
            model_uri=model_uri,
            name="SciDCC-Classifier",
        )

        # promoviraj u production samo ako je bolji od trenutnog
        client = mlflow.MlflowClient()
        # umjesto transition_model_version_stage, koristi aliases
        try:
            prod_version = client.get_model_version_by_alias("SciDCC-Classifier", "production")
            prod_f1 = client.get_run(prod_version.run_id).data.metrics.get("test_macro_f1", 0)
            if test_macro_f1 > prod_f1:
                client.set_registered_model_alias("SciDCC-Classifier", "production", registered.version)
                log.info("Novi model promoviran u production (F1: %.4f > %.4f)", test_macro_f1, prod_f1)
            else:
                client.set_registered_model_alias("SciDCC-Classifier", "staging", registered.version)
                log.info("Model ide u staging (F1: %.4f <= %.4f)", test_macro_f1, prod_f1)
        except Exception as e:
            # prvi model ili alias ne postoji – postavi production
            client.set_registered_model_alias("SciDCC-Classifier", "production", registered.version)
            log.info("Model registriran kao production (F1: %.4f)", test_macro_f1)
        except Exception as e:
            log.warning("MLflow registry promotion failed: %s", e)
            
        log.info("MLflow run zatvoren.")


if __name__ == "__main__":
    main()