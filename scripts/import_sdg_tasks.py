#!/usr/bin/env python3
"""
Import SDG annotation sentences into Label Studio as tasks.
Usage: python scripts/import_sdg_tasks.py
"""

import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN")
LABEL_STUDIO_PROJECT_ID = os.getenv("LABEL_STUDIO_PROJECT_ID", "1")
SENTENCES_FILE = "data/annotations/sdg_sentences.txt"


def main():
    if not LABEL_STUDIO_TOKEN:
        print("ERROR: LABEL_STUDIO_TOKEN nije postavljen u .env filu")
        return

    # učitaj rečenice
    sentences_path = Path(SENTENCES_FILE)
    if not sentences_path.exists():
        print(f"ERROR: File {SENTENCES_FILE} ne postoji.")
        print("Kopiraj sdg_annotation_sentences.txt u data/annotations/sdg_sentences.txt")
        return

    sentences = [
        line.strip()
        for line in sentences_path.read_text().splitlines()
        if line.strip()
    ]
    print(f"Učitano {len(sentences)} rečenica.")

    # formatiraj kao Label Studio tasks
    tasks = [{"data": {"text": sentence}} for sentence in sentences]

    # import u Label Studio
    headers = {
        "Authorization": f"Token {LABEL_STUDIO_TOKEN}",
        "Content-Type": "application/json",
    }

    url = f"{LABEL_STUDIO_URL}/api/projects/{LABEL_STUDIO_PROJECT_ID}/import"
    response = requests.post(url, headers=headers, json=tasks)

    if response.status_code in (200, 201):
        data = response.json()
        print(f"✅ Uspješno importano {data.get('task_count', len(tasks))} taskova u projekt {LABEL_STUDIO_PROJECT_ID}.")
        print(f"   Otvori Label Studio: {LABEL_STUDIO_URL}/projects/{LABEL_STUDIO_PROJECT_ID}/data")
    else:
        print(f"❌ Greška: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
