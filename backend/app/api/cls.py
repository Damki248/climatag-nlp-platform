from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.app.services.cls_service import cls_service

router = APIRouter(prefix="/api/cls", tags=["Classification"])


class CLSRequest(BaseModel):
    text: str = Field(..., max_length=50_000)
    top_k: int = 3


class CLSResponse(BaseModel):
    prediction: str
    score: float
    top_k: list


@router.post("/predict", response_model=CLSResponse)
def predict(request: CLSRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if not cls_service.available:
        raise HTTPException(status_code=503, detail="Classification model not available.")
    if request.top_k < 1 or request.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20.")
    result = cls_service.predict(request.text, top_k=request.top_k)
    return CLSResponse(**result)


@router.get("/status")
def status():
    return {"available": cls_service.available}