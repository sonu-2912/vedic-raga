from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.config import settings
from app.database import list_recommendation_history, save_recommendation_history
from app.schemas.recommendation import (
    ActivityRecommendationRequest,
    HistoryResponse,
    RecommendationResponse,
)
from app.services.auth_service import get_current_user
from app.services.environment_service import (
    get_environment_context,
    serialize_prahara_context,
    serialize_weather_context,
)
from app.services.mode_engine import is_activity_mode
from app.services.raga_recommender import RagaRecommender
from app.services.rasa_bhava_mapper import map_rasa_to_bhava
from app.services.rasa_model import RasaModelError, RasaPredictor

router = APIRouter(prefix="/api/recommend", tags=["recommendations"])


def _normalize_style(style: str) -> str:
    normalized = style.lower().strip()
    if normalized not in {"vocal", "instrumental", "both"}:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid style")
    return normalized


def _build_context(prahara_context, weather_context) -> dict:
    return {
        "prahara": prahara_context.prahara,
        "progress_in_prahara": prahara_context.progress_in_prahara,
        "arc": prahara_context.arc,
        "is_sandhya_kaal": prahara_context.is_sandhya_kaal,
        "sunrise": prahara_context.sunrise.isoformat(),
        "sunset": prahara_context.sunset.isoformat(),
        "ritu": weather_context.ritu,
        "weather_condition": weather_context.uddipana_condition,
        "temperature_c": weather_context.temperature_c,
        "humidity_pct": weather_context.humidity_pct,
        "wind_speed_kmh": weather_context.wind_speed_kmh,
        "wmo_code": weather_context.wmo_code,
        "weather_source": weather_context.source,
    }


def _save_history(
    *,
    user: dict,
    response: RecommendationResponse,
) -> None:
    save_recommendation_history(
        user_id=int(user["id"]),
        mode=response.mode,
        detected_rasa=response.detected_rasa,
        detected_bhava=response.detected_bhava,
        confidence=response.confidence,
        prahara=response.context.get("prahara"),
        ritu=response.context.get("ritu"),
        weather_condition=response.context.get("weather_condition"),
        recommendations=response.recommendations,
    )


@router.post("/image", response_model=RecommendationResponse)
async def recommend_from_image(
    request: Request,
    image: UploadFile = File(...),
    mode: str = Form(...),
    style: str = Form("both"),
    lat: float | None = Form(None),
    lon: float | None = Form(None),
    tz: str = Form(settings.default_tz),
    current_user: dict = Depends(get_current_user),
) -> RecommendationResponse:
    normalized_style = _normalize_style(style)
    predictor: RasaPredictor = request.app.state.rasa_predictor
    recommender: RagaRecommender = request.app.state.raga_recommender

    try:
        prediction = predictor.predict(image.file)
        detected_bhava = map_rasa_to_bhava(prediction.rasa)
        prahara_context, weather_context = await get_environment_context(lat, lon, tz)
        recommendations = recommender.recommend(
            mode=mode,
            detected_bhava=detected_bhava,
            current_prahara=prahara_context.prahara,
            current_ritu=weather_context.ritu,
            style=normalized_style,
        )
    except RasaModelError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    response = RecommendationResponse(
        mode=mode.lower().strip(),
        detected_rasa=prediction.rasa,
        detected_bhava=detected_bhava,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        low_confidence=prediction.confidence < settings.low_confidence_threshold,
        context=_build_context(prahara_context, weather_context),
        recommendations=[item.to_dict() for item in recommendations],
    )
    _save_history(user=current_user, response=response)
    return response


@router.post("/activity", response_model=RecommendationResponse)
async def recommend_activity(
    payload: ActivityRecommendationRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> RecommendationResponse:
    if not is_activity_mode(payload.mode):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Activity endpoint only supports study, gym, sleep, and meditation modes.",
        )

    normalized_style = _normalize_style(payload.style)
    recommender: RagaRecommender = request.app.state.raga_recommender
    try:
        prahara_context, weather_context = await get_environment_context(payload.lat, payload.lon, payload.tz)
        recommendations = recommender.recommend(
            mode=payload.mode,
            detected_bhava=None,
            current_prahara=prahara_context.prahara,
            current_ritu=weather_context.ritu,
            style=normalized_style,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    response = RecommendationResponse(
        mode=payload.mode.lower().strip(),
        detected_rasa=None,
        detected_bhava=None,
        confidence=None,
        probabilities={},
        low_confidence=False,
        context=_build_context(prahara_context, weather_context),
        recommendations=[item.to_dict() for item in recommendations],
    )
    _save_history(user=current_user, response=response)
    return response


@router.get("/history", response_model=HistoryResponse)
def history(current_user: dict = Depends(get_current_user)) -> HistoryResponse:
    return HistoryResponse(items=list_recommendation_history(int(current_user["id"])))

