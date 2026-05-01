from span_marker import SpanMarkerModel
from typing import List, Dict
from pathlib import Path
import torch

BASELINE_MODEL_PATH = "models/ner_baseline"
ADAPTED_MODEL_PATH  = "models/ner_adapted"


class NERService:
    def __init__(self):
        self.model = None
        self.active_model = None  # "baseline" ili "adapted"

    def load(self):
        """Učitava adapted model ako postoji, inače baseline."""
        if Path(ADAPTED_MODEL_PATH).exists():
            self._load_model(ADAPTED_MODEL_PATH, "adapted")
        else:
            self._load_model(BASELINE_MODEL_PATH, "baseline")

    def _load_model(self, path: str, name: str):
        print(f"Loading NER model from {path}...")
        self.model = SpanMarkerModel.from_pretrained(path)
        if torch.cuda.is_available():
            self.model.cuda()
        self.active_model = name
        print(f"NER model ready! (active: {name})")

    def switch_model(self, model_name: str):
        """Ručni switch između baseline i adapted modela."""
        if model_name == "adapted":
            if not Path(ADAPTED_MODEL_PATH).exists():
                raise FileNotFoundError(f"Adapted model not found at {ADAPTED_MODEL_PATH}")
            self._load_model(ADAPTED_MODEL_PATH, "adapted")
        elif model_name == "baseline":
            self._load_model(BASELINE_MODEL_PATH, "baseline")
        else:
            raise ValueError(f"Unknown model: {model_name}. Use 'baseline' or 'adapted'.")

    def predict(self, text: str) -> List[Dict]:
        if self.model is None:
            raise RuntimeError("Model not loaded!")
        entities = self.model.predict(text)
        return [
            {
                "span":  e["span"],
                "label": e["label"],
                "score": round(e["score"], 4),
                "start": e["char_start_index"],
                "end":   e["char_end_index"],
            }
            for e in entities
        ]

    def status(self) -> Dict:
        return {
            "active_model":       self.active_model,
            "baseline_available": Path(BASELINE_MODEL_PATH).exists(),
            "adapted_available":  Path(ADAPTED_MODEL_PATH).exists(),
        }


# Singleton – jedan model u memoriji
ner_service = NERService()