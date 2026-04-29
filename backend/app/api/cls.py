from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.app.services.cls_service import cls_service

router = APIRouter(prefix="/api/classify", tags=["Classification"])

class ClassifyRequest(BaseModel):
    text: str
    top_k: int = Field(default=3, ge=1, le=20)

class ClassifyResult(BaseModel):
    label: str
    score: float

class ClassifyResponse(BaseModel):
    label: str
    score: float
    top_k: list[ClassifyResult]

@router.post("/predict", response_model=ClassifyResponse)
def predict(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    result = cls_service.predict(request.text, top_k=request.top_k)
    return ClassifyResponse(
        label=result["label"],
        score=result["score"],
        top_k=[ClassifyResult(**r) for r in result["top_k"]],
    )