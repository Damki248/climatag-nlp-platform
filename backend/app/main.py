from fastapi import FastAPI
from contextlib import asynccontextmanager
from backend.app.api.ner import router as ner_router
from backend.app.api.annotation import router as annotation_router
from backend.app.api.cls import router as cls_router
from backend.app.services.ner_service import ner_service
from backend.app.services.cls_service import cls_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    ner_service.load()
    cls_service.load()
    yield

app = FastAPI(
    title="ClimaTag",
    description="NER and Text Classification for Climate Research",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(ner_router)
app.include_router(annotation_router)
app.include_router(cls_router)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}