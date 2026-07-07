from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from backend.app.services.label_studio_service import (
    upload_preannotated,
    export_annotations,
    ls_annotations_to_training_format,
)
from backend.app.services.ner_service import ner_service
import requests

router = APIRouter(prefix="/api/annotation", tags=["Annotation"])


class AnnotationEntity(BaseModel):
    span: str
    label: str
    start: int
    end: int


class AnnotationUploadRequest(BaseModel):
    texts: List[str]
    annotations: Optional[List[List[AnnotationEntity]]] = None


class AnnotationUploadResponse(BaseModel):
    uploaded: int
    project_id: int
    url: str


@router.post("/upload", response_model=AnnotationUploadResponse)
def upload(request: AnnotationUploadRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Max 100 texts per upload")
    try:
        annotations = None
        if request.annotations:
            annotations = [
                [e.model_dump() for e in ann_list]
                for ann_list in request.annotations
            ]
        result = upload_preannotated(request.texts, ner_service, annotations)
        return AnnotationUploadResponse(**result)
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Label Studio did not respond in time.")
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot reach Label Studio (is it running?)")
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Label Studio returned {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export")
def export(status: str = "completed"):
    try:
        tasks = export_annotations(status=status)
        training_data = ls_annotations_to_training_format(tasks)
        return {
            "task_count":     len(tasks),
            "training_samples": training_data,
        }
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Label Studio did not respond in time.")
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot reach Label Studio (is it running?)")
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Label Studio returned {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))