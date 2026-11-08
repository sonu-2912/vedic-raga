from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ActivityRecommendationRequest(BaseModel):
    mode: str
    style: str = "both"
    lat: float | None = None
    lon: float | None = None
    tz: str = "Asia/Kolkata"


class RecommendationResponse(BaseModel):
    mode: str
    detected_rasa: str | None
    detected_bhava: str | None
    confidence: float | None
    probabilities: dict[str, float] = Field(default_factory=dict)
    low_confidence: bool
    context: dict[str, Any]
    recommendations: list[dict[str, Any]]


class HistoryResponse(BaseModel):
    items: list[dict[str, Any]]

