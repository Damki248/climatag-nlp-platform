import json
import torch
from pathlib import Path
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import mlflow

FALLBACK_MODEL_PATH = "models/cls_full_ft_best"
LABEL_MAP_PATH = "data/processed/label_map.json"
REGISTRY_MODEL_NAME = "SciDCC-Classifier"

class ClassificationService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.id2label: Dict[int, str] = {}

    def load(self):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        self.id2label = {int(k): v for k, v in label_map["id2label"].items()}

        # pokušaj iz MLflow registrya
        model_path = self._resolve_model_path()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        if torch.cuda.is_available():
            self.model.cuda()

        print(f"Classification model ready! ({len(self.id2label)} classes)")

    def _resolve_model_path(self) -> str:
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            client = mlflow.MlflowClient()
            # koristi aliases umjesto stages (MLflow 2.9+)
            version = client.get_model_version_by_alias(REGISTRY_MODEL_NAME, "production")
            local_path = mlflow.artifacts.download_artifacts(
                run_id=version.run_id,
                artifact_path="cls_model",
            )
            print(f"Loaded model from MLflow Registry (run: {version.run_id[:8]}...)")
            return local_path
        except Exception as e:
            print(f"MLflow registry not available ({e}), falling back to local path.")

        print(f"Loading classification model from {FALLBACK_MODEL_PATH}...")
        return FALLBACK_MODEL_PATH

    def predict(self, text: str, top_k: int = 3) -> Dict:
        if self.model is None:
            raise RuntimeError("Model not loaded!")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0].cpu()
        top_indices = probs.argsort(descending=True)[:top_k].tolist()

        return {
            "label": self.id2label[top_indices[0]],
            "score": round(float(probs[top_indices[0]]), 4),
            "top_k": [
                {
                    "label": self.id2label[i],
                    "score": round(float(probs[i]), 4),
                }
                for i in top_indices
            ],
        }

# singleton
cls_service = ClassificationService()