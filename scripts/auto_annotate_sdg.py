#!/usr/bin/env python3
"""
Auto-anotacija SDG uzoraka i upload u Label Studio.

Skripta:
1. Učitava rečenice iz tekstualnog fajla (jedna po retku)
2. Detektira SDG pojmove exact matchom (case-insensitive)
3. Uploadava taskove u Label Studio s pre-postavljenim SDG anotacijama
4. Ti samo prolazis kroz Label Studio i klikaš Submit

Usage:
    python scripts/auto_annotate_sdg.py --sentences data/annotations/sdg_sentences_v2.txt
"""

import os
import json
import uuid
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN")
LABEL_STUDIO_PROJECT_ID = os.getenv("LABEL_STUDIO_PROJECT_ID", "1")

# Lista SDG pojmova za exact match detekciju
SDG_TERMS = [
    # SDG 1
    "no poverty",
    # SDG 2
    "zero hunger", "food security",
    # SDG 3
    "good health and well-being", "good health", "well-being",
    # SDG 4
    "quality education", "inclusive education",
    # SDG 5
    "gender equality", "women empowerment",
    # SDG 6
    "clean water", "clean water and sanitation", "safe drinking water", "water sanitation",
    # SDG 7
    "affordable and clean energy", "clean energy", "renewable energy access",
    # SDG 8
    "decent work", "economic growth", "decent work and economic growth",
    # SDG 9
    "industry innovation and infrastructure", "resilient infrastructure", "sustainable industrialization",
    # SDG 10
    "reduced inequalities", "reduced inequality", "income inequality",
    # SDG 11
    "sustainable cities", "sustainable communities", "sustainable urban",
    # SDG 12
    "responsible consumption and production", "responsible consumption", "sustainable consumption",
    # SDG 13
    "climate action", "climate change action",
    # SDG 14
    "life below water", "ocean conservation", "marine conservation",
    # SDG 15
    "life on land", "terrestrial ecosystems", "sustainable forests",
    # SDG 16
    "peace justice and strong institutions", "peace and justice", "strong institutions",
    # SDG 17
    "partnerships for the goals", "global partnership", "partnerships for sustainable development",
]

# Sortiraj od najduljeg prema najkraćem da bi duži matchevi imali prioritet
SDG_TERMS_SORTED = sorted(SDG_TERMS, key=len, reverse=True)


def find_sdg_entities(text: str) -> list[dict]:
    """
    Pronalazi SVE SDG pojmove u tekstu (case-insensitive exact match).
    Vraća listu {start, end, text} bez preklapanja.
    """
    text_lower = text.lower()
    found = []
    covered = set()  # skup indeksa koji su već pokriveni

    for term in SDG_TERMS_SORTED:
        start = 0
        while True:
            idx = text_lower.find(term, start)
            if idx == -1:
                break
            end = idx + len(term)
            span_indices = set(range(idx, end))
            # preskači ako se preklapa s već pronađenim entitetom
            if not span_indices & covered:
                found.append({
                    "start": idx,
                    "end": end,
                    "text": text[idx:end],
                })
                covered |= span_indices
            start = idx + 1

    return sorted(found, key=lambda x: x["start"])


def build_ls_task(text: str, entities: list[dict]) -> dict:
    """
    Gradi Label Studio task s pre-postavljenim SDG anotacijama kao predictions.
    """
    predictions = []
    for entity in entities:
        predictions.append({
            "id": str(uuid.uuid4())[:8],
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "origin": "manual",  # označeno kao manual da se pojavi kao anotacija
            "value": {
                "start": entity["start"],
                "end": entity["end"],
                "text": entity["text"],
                "labels": ["SDG"],
            },
        })

    return {
        "data": {"text": text},
        "annotations": [
            {
                "result": predictions,
                "was_cancelled": False,
                "ground_truth": False,
            }
        ] if predictions else [],
    }


def upload_tasks(tasks: list[dict]) -> bool:
    """Upload taskova u Label Studio."""
    headers = {
        "Authorization": f"Token {LABEL_STUDIO_TOKEN}",
        "Content-Type": "application/json",
    }
    url = f"{LABEL_STUDIO_URL}/api/projects/{LABEL_STUDIO_PROJECT_ID}/import"
    response = requests.post(url, headers=headers, json=tasks)

    if response.status_code in (200, 201):
        data = response.json()
        print(f"✅ Uploadano {data.get('task_count', len(tasks))} taskova.")
        return True
    else:
        print(f"❌ Greška {response.status_code}: {response.text}")
        return False


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sentences", default="data/annotations/sdg_sentences_v2.txt",
                   help="Tekstualni file s rečenicama (jedna po retku)")
    p.add_argument("--dry_run", action="store_true",
                   help="Samo prikaži što bi se uploadalo, bez stvarnog uploada")
    p.add_argument("--min_entities", default=1, type=int,
                   help="Minimalan broj SDG entiteta po rečenici (default: 1)")
    args = p.parse_args()

    if not LABEL_STUDIO_TOKEN:
        print("ERROR: LABEL_STUDIO_TOKEN nije postavljen u .env")
        return

    sentences_path = Path(args.sentences)
    if not sentences_path.exists():
        print(f"ERROR: File {args.sentences} ne postoji.")
        return

    sentences = [
        line.strip()
        for line in sentences_path.read_text().splitlines()
        if line.strip()
    ]
    print(f"Učitano {len(sentences)} rečenica.")

    tasks = []
    skipped = 0
    total_entities = 0

    for sentence in sentences:
        entities = find_sdg_entities(sentence)
        if len(entities) < args.min_entities:
            skipped += 1
            continue
        total_entities += len(entities)
        tasks.append(build_ls_task(sentence, entities))

    print(f"Pronađeno {len(tasks)} rečenica s SDG entitetima ({total_entities} ukupno).")
    print(f"Preskočeno {skipped} rečenica bez SDG entiteta.")

    if args.dry_run:
        print("\n--- DRY RUN – prvih 5 taskova ---")
        for task in tasks[:5]:
            text = task["data"]["text"]
            anns = task["annotations"][0]["result"] if task["annotations"] else []
            spans = [a["value"]["text"] for a in anns]
            print(f"  TEXT: {text[:80]}")
            print(f"  SDG:  {spans}")
            print()
        return

    upload_tasks(tasks)


if __name__ == "__main__":
    main()
