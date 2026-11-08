from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    backend_root: Path
    repo_root: Path
    database_path: Path
    raga_data_path: Path
    model_path: Path
    secret_key: str
    access_token_minutes: int
    cors_origins: list[str]
    default_lat: float
    default_lon: float
    default_tz: str
    low_confidence_threshold: float


def _split_origins(value: str) -> list[str]:
    return [origin.strip() for origin in value.split(",") if origin.strip()]


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent

settings = Settings(
    backend_root=BACKEND_ROOT,
    repo_root=REPO_ROOT,
    database_path=Path(os.getenv("DATABASE_PATH", BACKEND_ROOT / "local_dev.db")),
    raga_data_path=Path(
        os.getenv("RAGA_DATA_PATH", REPO_ROOT / "raga_sthayi_bhava_dataset_enriched.json")
    ),
    model_path=Path(os.getenv("RASA_MODEL_PATH", REPO_ROOT / "best_navarasa_model.pth")),
    secret_key=os.getenv("APP_SECRET_KEY", "local-dev-change-me"),
    access_token_minutes=int(os.getenv("ACCESS_TOKEN_MINUTES", "10080")),
    cors_origins=_split_origins(
        os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    ),
    default_lat=float(os.getenv("DEFAULT_LAT", "28.6139")),
    default_lon=float(os.getenv("DEFAULT_LON", "77.2090")),
    default_tz=os.getenv("DEFAULT_TZ", "Asia/Kolkata"),
    low_confidence_threshold=float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.55")),
)

