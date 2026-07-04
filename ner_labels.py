"""Canonical NER label set for ClimaTag.

Single source of truth for the entity categories the GLiNER model works with:
the 28 CliReNER categories plus the custom Climate Model category.

Imported by:
- backend/app/services/ner_service.py  (inference)
- training/ner_gliner/train.py         (fine-tuning + evaluation)

The frontend keeps a mirror of this list in
frontend/src/constants/nerLabels.js, keeps the two in sync.
"""

NER_LABELS = [
    "Asset", "Body Part", "Body of Water", "Chemical", "Disease",
    "Ecosystem", "Energy Source", "Field of Study", "Geographical Feature",
    "Intellectual Artefact", "Location", "Mathematical Expression",
    "Measuring Device", "Meteorological Phenomenon", "Method",
    "Natural Disaster", "Natural Phenomenon", "Organism", "Organization",
    "Other", "Person", "Physical Artefact", "Physical Phenomenon",
    "Policy", "Quantity", "Satellite", "System", "Time Period",
    "Climate Model",
]