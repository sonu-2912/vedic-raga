from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth_routes import router as auth_router
from app.api.mode_routes import router as mode_router
from app.api.recommendation_routes import router as recommendation_router
from app.config import settings
from app.database import init_db
from app.services.raga_recommender import RagaRecommender
from app.services.raga_repository import RagaRepository
from app.services.rasa_model import RasaPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    repository = RagaRepository(settings.raga_data_path)
    app.state.raga_repository = repository
    app.state.raga_recommender = RagaRecommender(repository)
    app.state.rasa_predictor = RasaPredictor(settings.model_path)
    yield


app = FastAPI(
    title="Raga Recommender Local API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(mode_router)
app.include_router(recommendation_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

