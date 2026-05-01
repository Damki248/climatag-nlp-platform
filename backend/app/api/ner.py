from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.services.ner_service import ner_service

router = APIRouter(prefix="/api/ner", tags=["NER"])


class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    entities: list
    count: int
    model: str

class SwitchRequest(BaseModel):
    model: str  # "baseline" ili "adapted"


@router.post("/predict", response_model=NERResponse)
def predict(request: NERRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    entities = ner_service.predict(request.text)
    return NERResponse(
        entities=entities,
        count=len(entities),
        model=ner_service.active_model,
    )

@router.get("/status")
def status():
    return ner_service.status()

@router.post("/switch")
def switch(request: SwitchRequest):
    try:
        ner_service.switch_model(request.model)
        return {"message": f"Switched to {request.model} model", "active_model": request.model}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))