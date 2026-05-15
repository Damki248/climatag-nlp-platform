# backend/app/api/train.py
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/train", tags=["Training"])

# Status file za praćenje progresa
STATUS_FILE = Path("training_status.json")


class TrainRequest(BaseModel):
    epochs:       int   = 10
    lr:           float = 5e-6
    batch:        int   = 8
    silver_ratio: int   = 5
    run_name:     Optional[str] = None


class TrainStatus(BaseModel):
    status:    str   # idle, running, finished, failed
    run_name:  Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    current_epoch: Optional[int] = None
    total_epochs:  Optional[int] = None
    logs:      list  = []
    error:     Optional[str] = None


def _write_status(data: dict):
    STATUS_FILE.write_text(json.dumps(data))


def _read_status() -> dict:
    if not STATUS_FILE.exists():
        return {"status": "idle", "logs": []}
    return json.loads(STATUS_FILE.read_text())


def _run_training(request: TrainRequest):
    """Pokreće trening kao subprocess i prati output."""
    run_name = request.run_name or f"gliner_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    _write_status({
        "status": "running",
        "run_name": run_name,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "current_epoch": 0,
        "total_epochs": request.epochs,
        "logs": [f"Starting training run: {run_name}"],
        "error": None,
    })

    cmd = [
        sys.executable, "-m", "training.ner_gliner.train",
        "--cm_annotations",  "data/annotations/climate_model_annotations.json",
        "--base_model",      "models/ner_gliner_baseline",
        "--output_model",    "models/ner_gliner_climate_model",
        "--epochs",          str(request.epochs),
        "--lr",              str(request.lr),
        "--batch",           str(request.batch),
        "--silver_ratio",    str(request.silver_ratio),
        "--run_name",        run_name,
        "--experiment",      "climtag_ner_gliner",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        logs = [f"Starting training run: {run_name}"]
        current_epoch = 0

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            logs.append(line)
            # keep last 50 log lines
            if len(logs) > 50:
                logs = logs[-50:]

            # parsaj epohu iz loga
            if "epoch" in line.lower():
                for word in line.split():
                    try:
                        val = float(word.strip(",'"))
                        if 1 <= val <= request.epochs:
                            current_epoch = int(val)
                            break
                    except ValueError:
                        pass

            _write_status({
                "status": "running",
                "run_name": run_name,
                "started_at": _read_status().get("started_at"),
                "finished_at": None,
                "current_epoch": current_epoch,
                "total_epochs": request.epochs,
                "logs": logs,
                "error": None,
            })

        proc.wait()

        if proc.returncode == 0:
            logs.append("✅ Training completed successfully!")
            _write_status({
                "status": "finished",
                "run_name": run_name,
                "started_at": _read_status().get("started_at"),
                "finished_at": datetime.now().isoformat(),
                "current_epoch": request.epochs,
                "total_epochs": request.epochs,
                "logs": logs,
                "error": None,
            })
        else:
            _write_status({
                "status": "failed",
                "run_name": run_name,
                "started_at": _read_status().get("started_at"),
                "finished_at": datetime.now().isoformat(),
                "current_epoch": current_epoch,
                "total_epochs": request.epochs,
                "logs": logs,
                "error": f"Process exited with code {proc.returncode}",
            })

    except Exception as e:
        _write_status({
            "status": "failed",
            "run_name": run_name,
            "started_at": _read_status().get("started_at"),
            "finished_at": datetime.now().isoformat(),
            "current_epoch": 0,
            "total_epochs": request.epochs,
            "logs": [str(e)],
            "error": str(e),
        })


@router.post("/start")
def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    status = _read_status()
    if status.get("status") == "running":
        raise HTTPException(status_code=409, detail="Training already in progress")

    background_tasks.add_task(_run_training, request)
    return {"message": "Training started", "run_name": request.run_name}


@router.get("/status", response_model=TrainStatus)
def get_status():
    return TrainStatus(**_read_status())


@router.post("/reset")
def reset_status():
    """Reset statusa nakon failed/finished runa."""
    status = _read_status()
    if status.get("status") == "running":
        raise HTTPException(status_code=409, detail="Cannot reset while training is running")
    _write_status({"status": "idle", "logs": []})
    return {"message": "Status reset"}