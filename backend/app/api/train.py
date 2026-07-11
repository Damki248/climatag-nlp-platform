# backend/app/api/train.py
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

_start_lock = threading.Lock()

router = APIRouter(prefix="/api/train", tags=["Training"])

# Status file for project tracking
STATUS_FILE = Path("training_status.json")


class TrainRequest(BaseModel):
    epochs:       int   = Field(10, ge=1, le=50)
    lr:           float = Field(5e-6, gt=0, le=1e-2)
    batch:        int   = Field(8, ge=1, le=64)
    silver_ratio: int   = Field(5, ge=1, le=20)
    run_name:     Optional[str] = Field(None, max_length=100, pattern=r"^[\w\-.]+$")


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
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATUS_FILE)


def _read_status() -> dict:
    if not STATUS_FILE.exists():
        return {"status": "idle", "logs": []}
    status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    
    if status.get("status") == "running" and status.get("pid"):
        try:
            os.kill(status["pid"], 0)
        except:
            status["staus"] = "failed"
            status["finished_at"] = datetime.now().isoformat()
            status["error"] = "Training process died unexpectedly (backend restart?)"
            _write_status(status)
    return status


def _run_training(request: TrainRequest, run_name: str):
    """Starts the training as a subprocess and tracks the output."""

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

        status = _read_status()
        status["pid"] = proc.pid
        _write_status(status)

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

            # parse the epoch from log file
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
                "pid": proc.pid,
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
    with _start_lock:
        if _read_status().get("status") == "running":
            raise HTTPException(status_code=409, detail="Training already in progress")
        run_name = request.run_name or f"gliner_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        _write_status({"status": "running", "run_name": run_name, "logs": ["Queued..."],
                       "started_at": datetime.now().isoformat()})
        background_tasks.add_task(_run_training, request, run_name)
        return {"message": "Training started", "run_name": run_name}


@router.get("/status", response_model=TrainStatus)
def get_status():
    return TrainStatus(**_read_status())


@router.post("/reset")
def reset_status():
    """Reset status after failed/finished run"""
    status = _read_status()
    if status.get("status") == "running":
        raise HTTPException(status_code=409, detail="Cannot reset while training is running")
    _write_status({"status": "idle", "logs": []})
    return {"message": "Status reset"}
