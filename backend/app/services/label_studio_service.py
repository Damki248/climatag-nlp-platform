import json
import os
import uuid
import requests
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv(override=True)

LS_URL     = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LS_TOKEN   = os.getenv("LABEL_STUDIO_TOKEN")
LS_PROJECT = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))

HEADERS = {
    "Authorization": f"Token {LS_TOKEN}",
    "Content-Type": "application/json",
}


def _ner_to_ls_annotations(text: str, entities: List[Dict]) -> List[Dict]:
    """Konvertira NER output u Label Studio annotation format."""
    results = []
    for e in entities:
        results.append({
            "id": str(uuid.uuid4())[:8],
            "type": "labels",
            "from_name": "label",
            "to_name": "text",
            "value": {
                "start":  e["start"],
                "end":    e["end"],
                "text":   e["span"],
                "labels": [e["label"]],
            },
        })
    return results


def upload_preannotated(texts: List[str], ner_service, annotations: List[List[Dict]] = None) -> Dict:
    """
    Prima listu tekstova i opcionalne human anotacije.
    Ako su annotations None, koristi NER model za pre-anotaciju.
    Ako su proslijeđene, koristi ih direktno (human-corrected).
    """
    tasks = []
    for i, text in enumerate(texts):
        is_human = annotations and i < len(annotations) and annotations[i]

        if is_human:
            entities = annotations[i]
            result_annotations = [
                {
                    "id": f"ent_{j}",
                    "type": "labels",
                    "from_name": "label",
                    "to_name": "text",
                    "value": {
                        "start":  e["start"],
                        "end":    e["end"],
                        "text":   e["span"],
                        "labels": [e["label"]],
                    },
                }
                for j, e in enumerate(entities)
            ]
            score = 1.0
            model_version = "human-corrected"
        else:
            predicted = ner_service.predict(text)
            result_annotations = _ner_to_ls_annotations(text, predicted)
            score = round(
                sum(e["score"] for e in predicted) / len(predicted), 4
            ) if predicted else 0.0
            model_version = "GLiNER-climate-model"

        task = {
            "data": {"text": text},
            "predictions": [
                {
                    "model_version": model_version,
                    "score": score,
                    "result": result_annotations,
                }
            ],
        }
        tasks.append(task)

    url = f"{LS_URL}/api/projects/{LS_PROJECT}/import"
    response = requests.post(url, headers=HEADERS, json=tasks)
    response.raise_for_status()

    result = response.json()
    return {
        "uploaded": result.get("task_count", len(tasks)),
        "project_id": LS_PROJECT,
        "url": f"{LS_URL}/projects/{LS_PROJECT}/data",
    }


def export_annotations(status: str = "completed") -> List[Dict]:
    """Exporta anotacije iz Label Studio projekta."""
    url = f"{LS_URL}/api/projects/{LS_PROJECT}/export"
    params = {"exportType": "JSON"}
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()

    tasks = response.json()

    if status == "completed":
        tasks = [t for t in tasks if t.get("annotations")]

    return tasks


def ls_annotations_to_training_format(tasks: List[Dict]) -> List[Dict]:
    """Konvertira Label Studio export u GLiNER training format."""
    training_samples = []
    for task in tasks:
        text = task["data"]["text"]
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        latest = sorted(annotations, key=lambda a: a["created_at"])[-1]
        spans = latest.get("result", [])

        entities = []
        for span in spans:
            val = span.get("value", {})
            labels = val.get("labels", [])
            if not labels:
                continue
            entities.append({
                "span":  val.get("text", ""),
                "label": labels[0],
                "start": val.get("start"),
                "end":   val.get("end"),
            })

        training_samples.append({
            "text":     text,
            "entities": entities,
        })

    return training_samples