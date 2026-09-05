from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from vision_language_clothing_retrieval.api.routes import router
from vision_language_clothing_retrieval.api.fake_retrieval_service import (
    RetrievalService,
)

MODEL_PATH = "embeddings/multimodal_model.pt"
EMBEDDINGS_PATH = "embeddings/test.pt"
IMAGES_DIR = "data/images"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retrieval_service = RetrievalService(
        MODEL_PATH,
        EMBEDDINGS_PATH
    )
    yield

app = FastAPI(
    title="Vision Language Clothing Retrieval API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if Path(IMAGES_DIR).exists():
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

app.include_router(router)