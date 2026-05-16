from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
import os

from backend.app.api.ner import router as ner_router
from backend.app.api.annotation import router as annotation_router
from backend.app.services.ner_service import ner_service
from backend.app.api.train import router as train_router
from backend.app.api.cls import router as cls_router
from backend.app.services.cls_service import cls_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    ner_service.load()
    cls_service.load()
    yield


app = FastAPI(
    title="ClimaTag",
    description="Named Entity Recognition for Climate Research",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS – u produkciji postavi ALLOWED_ORIGINS u .env
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ner_router)
app.include_router(annotation_router)
app.include_router(train_router)
app.include_router(cls_router)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.2.0"}


# Serviranje React frontend builda (samo u produkciji)
DIST_DIR = Path("frontend/dist")
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str):
        # API rute preskačemo
        if full_path.startswith("api/"):
            return {"detail": "Not found"}
        index = DIST_DIR / "index.html"
        return FileResponse(str(index))