from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.services.ner_service import ner_service

router = APIRouter(prefix="/api/ner", tags=["NER"])

class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    entities: list
    count: int

@router.post("/predict", response_model=NERResponse)
def predict(request: NERRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    entities = ner_service.predict(request.text)
    return NERResponse(entities=entities, count=len(entities))