from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import torch
import logging
import os

log = logging.getLogger(__name__)

CLS_MODEL_PATH = os.getenv("CLS_MODEL_PATH", "models/cls_full_ft_body_lr2e5_batch16_exp07/checkpoint-11110")


class CLSService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self):
        if not Path(CLS_MODEL_PATH).exists():
            log.warning("Classification model not found at %s — service disabled.", CLS_MODEL_PATH)
            return
        log.info("Loading classification model from %s...", CLS_MODEL_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_PATH)
        self.model.to(self.device)
        self.model.eval()
        log.info("Classification model ready! (device: %s)", self.device)

    def predict(self, text: str, top_k: int = 3) -> dict:
        if self.model is None:
            raise RuntimeError("Classification model not loaded.")
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        ranked = sorted(
            [{"label": self.model.config.id2label[i], "score": round(p, 4)} for i, p in enumerate(probs)],
            key=lambda x: x["score"],
            reverse=True,
        )
        return {
            "prediction": ranked[0]["label"],
            "score": ranked[0]["score"],
            "top_k": ranked[:top_k],
        }

    @property
    def available(self) -> bool:
        return self.model is not None


# Singleton
cls_service = CLSService()