from span_marker import SpanMarkerModel
from typing import List, Dict
import torch

MODEL_PATH = "models/ner_baseline"

class NERService:
    def __init__(self):
        self.model = None

    def load(self):
        print(f"Loading NER model from {MODEL_PATH}...")
        self.model = SpanMarkerModel.from_pretrained(MODEL_PATH)
        if torch.cuda.is_available():
            self.model.cuda()
        print("NER model ready!")

    def predict(self, text: str) -> List[Dict]:
        if self.model is None:
            raise RuntimeError("Model not loaded!")
        entities = self.model.predict(text)
        return [
            {
                "span": e["span"],
                "label": e["label"],
                "score": round(e["score"], 4),
                "start": e["char_start_index"],
                "end": e["char_end_index"],
            }
            for e in entities
        ]

# Singleton – jedan model u memoriji
ner_service = NERService()