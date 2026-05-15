from fastapi import FastAPI
from contextlib import asynccontextmanager
from backend.app.api.ner import router as ner_router
from backend.app.api.annotation import router as annotation_router
from backend.app.services.ner_service import ner_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    ner_service.load()
    yield


app = FastAPI(
    title="ClimaTag",
    description="Named Entity Recognition for Climate Research",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(ner_router)
app.include_router(annotation_router)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.2.0"}