# eval_baseline.py — one-off, ne treba u repo
from gliner import GLiNER
import json, random
from training.ner_gliner.train import parse_climate_model_annotations, evaluate_climate_model
from transformers import set_seed

set_seed(42)   # isti seed => isti held-out split kao u treningu
cm = parse_climate_model_annotations("data/annotations/climate_model_annotations.json")
random.shuffle(cm)
n_test = max(10, int(len(cm) * 0.20))
cm_test = cm[:n_test]

model = GLiNER.from_pretrained("models/ner_gliner_baseline")
print(evaluate_climate_model(model, cm_test, n=len(cm_test)))