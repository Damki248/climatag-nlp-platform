from gliner import GLiNER
from typing import List, Dict
from pathlib import Path
import torch

GLINER_MODEL_PATH    = "models/ner_gliner_climate_model"
GLINER_BASELINE_PATH = "models/ner_gliner_baseline"

GLINER_LABELS = [
    "Asset", "Body Part", "Body of Water", "Chemical", "Disease",
    "Ecosystem", "Energy Source", "Field of Study", "Geographical Feature",
    "Intellectual Artefact", "Location", "Meteorological Phenomenon", "Method",
    "Natural Disaster", "Natural Phenomenon", "Organism", "Organization",
    "Other", "Person", "Physical Artefact", "Physical Phenomenon",
    "Policy", "Quantity", "Satellite", "System", "Time Period", "Climate Model",
]


class NERService:
    def __init__(self):
        self.model = None
        self.active_model = None  # "baseline" ili "climate_model"

    def load(self):
        """Učitava fine-tuned Climate Model model ako postoji, inače baseline."""
        if Path(GLINER_MODEL_PATH).exists():
            self._load_model(GLINER_MODEL_PATH, "climate_model")
        else:
            self._load_model(GLINER_BASELINE_PATH, "baseline")

    def _load_model(self, path: str, name: str):
        print(f"Loading GLiNER model from {path}...")
        self.model = GLiNER.from_pretrained(path)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.active_model = name
        print(f"GLiNER model ready! (active: {name})")

    def switch_model(self, model_name: str):
        """Switch između baseline i climate_model."""
        if model_name == "climate_model":
            if not Path(GLINER_MODEL_PATH).exists():
                raise FileNotFoundError(f"Climate Model not found at {GLINER_MODEL_PATH}")
            self._load_model(GLINER_MODEL_PATH, "climate_model")
        elif model_name == "baseline":
            self._load_model(GLINER_BASELINE_PATH, "baseline")
        else:
            raise ValueError(f"Unknown model: {model_name}. Use 'baseline' or 'climate_model'.")

    def predict(self, text: str) -> List[Dict]:
        if self.model is None:
            raise RuntimeError("GLiNER model not loaded!")
        entities = self.model.predict_entities(text, GLINER_LABELS)
        return [
            {
                "span":  e["text"],
                "label": e["label"],
                "score": round(e["score"], 4),
                "start": e["start"],
                "end":   e["end"],
            }
            for e in entities
        ]

    def status(self) -> Dict:
        return {
            "active_model":            self.active_model,
            "baseline_available":      Path(GLINER_BASELINE_PATH).exists(),
            "climate_model_available": Path(GLINER_MODEL_PATH).exists(),
        }


# Singleton
ner_service = NERService()