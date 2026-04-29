import json
import torch
from pathlib import Path
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/cls_full_ft_best"
LABEL_MAP_PATH = "data/processed/label_map.json"

class ClassificationService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.id2label: Dict[int, str] = {}

    def load(self):
        print(f"Loading classification model from {MODEL_PATH}...")

        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        self.id2label = {int(k): v for k, v in label_map["id2label"].items()}

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.eval()

        if torch.cuda.is_available():
            self.model.cuda()

        print(f"Classification model ready! ({len(self.id2label)} classes)")

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