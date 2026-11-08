from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from app.config import settings

log = logging.getLogger(__name__)

SANDHYA_KAAL_MINUTES = 45

_WMO_UDDIPANA_MAP: dict[range | int, str] = {
    0: "clear_sky",
    1: "clear_sky",
    2: "overcast",
    3: "overcast",
    range(4, 13): "clear_sky",
    range(13, 20): "overcast",
    range(20, 30): "rain",
    range(30, 40): "stormy",
    range(40, 50): "fog",
    50: "rain",
    range(51, 68): "rain",
    range(68, 70): "cold",
    range(71, 78): "cold",
    range(78, 80): "cold",
    range(80, 83): "monsoon",
    range(83, 85): "rain",
    range(85, 87): "cold",
    range(87, 91): "cold",
    range(91, 95): "rain",
    range(95, 100): "stormy",
}

_RITU_MAP: list[tuple[tuple[int, int], str]] = [
    ((1, 2), "shishira"),
    ((3, 4), "vasanta"),
    ((5, 6), "grishma"),
    ((7, 8), "varsha"),
    ((9, 10), "sharad"),
    ((11, 12), "hemanta"),
]


@dataclass(frozen=True, slots=True)
class PraharaContext:
    prahara: int
    progress_in_prahara: float
    is_sandhya_kaal: bool
    arc: str
    sunrise: datetime
    sunset: datetime


@dataclass(frozen=True, slots=True)
class WeatherContext:
    wmo_code: int | None
    uddipana_condition: str
    temperature_c: float | None
    humidity_pct: int | None
    wind_speed_kmh: float | None
    ritu: str
    is_precipitation_likely: bool
    source: str


def _resolve_wmo_condition(code: int) -> str:
    for key, condition in _WMO_UDDIPANA_MAP.items():
        if isinstance(key, range):
            if code in key:
                return condition
        elif key == code:
            return condition
    return "unknown"


def determine_ritu(month: int) -> str:
    for (month_start, month_end), ritu in _RITU_MAP:
        if month_start <= month <= month_end:
            return ritu
    raise ValueError(f"Month {month} out of range 1-12.")


def _prahara_from_arc(
    now: datetime,
    arc_start: datetime,
    arc_end: datetime,
    arc_offset: int,
) -> tuple[int, float]:
    arc_duration = (arc_end - arc_start).total_seconds()
    prahara_duration = arc_duration / 4.0
    elapsed = max(0.0, min((now - arc_start).total_seconds(), arc_duration - 1))
    slot = int(elapsed // prahara_duration)
    progress = (elapsed % prahara_duration) / prahara_duration
    return arc_offset + slot + 1, round(progress, 4)


def _build_prahara_context(
    now: datetime,
    sunrise_today: datetime,
    sunset_today: datetime,
    sunrise_tomorrow: datetime,
    sunset_yesterday: datetime,
) -> PraharaContext:
    sandhya_delta = timedelta(minutes=SANDHYA_KAAL_MINUTES)
    is_sandhya_kaal = abs(now - sunrise_today) < sandhya_delta or abs(now - sunset_today) < sandhya_delta

    if sunrise_today <= now < sunset_today:
        prahara, progress = _prahara_from_arc(now, sunrise_today, sunset_today, 0)
        arc = "day"
    elif now >= sunset_today:
        prahara, progress = _prahara_from_arc(now, sunset_today, sunrise_tomorrow, 4)
        arc = "night"
    else:
        prahara, progress = _prahara_from_arc(now, sunset_yesterday, sunrise_today, 4)
        arc = "night"

    return PraharaContext(
        prahara=prahara,
        progress_in_prahara=progress,
        is_sandhya_kaal=is_sandhya_kaal,
        arc=arc,
        sunrise=sunrise_today,
        sunset=sunset_today,
    )


def calculate_prahara(lat: float, lon: float, tz_str: str) -> PraharaContext:
    try:
        tz = ZoneInfo(tz_str)
    except ZoneInfoNotFoundError as exc:
        raise ZoneInfoNotFoundError(f"Unknown timezone: '{tz_str}'") from exc

    now = datetime.now(tz=tz)
    local_date = now.date()
    sunrise_today = datetime.combine(local_date, time(hour=6), tzinfo=tz)
    sunset_today = datetime.combine(local_date, time(hour=18), tzinfo=tz)
    sunrise_tomorrow = sunrise_today + timedelta(days=1)
    sunset_yesterday = sunset_today - timedelta(days=1)
    return _build_prahara_context(
        now=now,
        sunrise_today=sunrise_today,
        sunset_today=sunset_today,
        sunrise_tomorrow=sunrise_tomorrow,
        sunset_yesterday=sunset_yesterday,
    )


def _fallback_weather_context(tz_str: str) -> WeatherContext:
    month = datetime.now(tz=ZoneInfo(tz_str)).month
    return WeatherContext(
        wmo_code=None,
        uddipana_condition="unknown",
        temperature_c=None,
        humidity_pct=None,
        wind_speed_kmh=None,
        ritu=determine_ritu(month),
        is_precipitation_likely=False,
        source="fallback",
    )


async def _fetch_open_meteo_payload(lat: float, lon: float, tz_str: str) -> dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "hourly": "precipitation_probability,relative_humidity_2m",
        "daily": "sunrise,sunset",
        "past_days": 1,
        "forecast_days": 2,
        "timezone": tz_str,
    }
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        return response.json()


def _build_weather_context(data: dict[str, Any], tz_str: str) -> WeatherContext:
    current_weather = data["current_weather"]
    wmo_code = int(current_weather["weathercode"])
    hourly = data.get("hourly", {})
    humidity_list = hourly.get("relative_humidity_2m", [])
    precip_probs = hourly.get("precipitation_probability", [])
    weather_tz_name = data.get("timezone") or tz_str
    try:
        weather_tz = ZoneInfo(weather_tz_name)
    except ZoneInfoNotFoundError:
        weather_tz = ZoneInfo(tz_str)
    return WeatherContext(
        wmo_code=wmo_code,
        uddipana_condition=_resolve_wmo_condition(wmo_code),
        temperature_c=float(current_weather["temperature"]),
        humidity_pct=int(humidity_list[0]) if humidity_list else None,
        wind_speed_kmh=float(current_weather["windspeed"]),
        ritu=determine_ritu(datetime.now(tz=weather_tz).month),
        is_precipitation_likely=any(int(p) >= 50 for p in precip_probs[:3]),
        source="open-meteo",
    )


def _parse_local_datetime(value: str, tz_str: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ZoneInfo(tz_str))
    return parsed.astimezone(ZoneInfo(tz_str))


def _build_live_prahara_context(data: dict[str, Any], tz_str: str) -> PraharaContext | None:
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    sunrises = daily.get("sunrise", [])
    sunsets = daily.get("sunset", [])
    if not dates or not sunrises or not sunsets:
        return None

    tz = ZoneInfo(tz_str)
    now = datetime.now(tz=tz)
    by_date = {
        datetime.fromisoformat(date_value).date(): {
            "sunrise": _parse_local_datetime(sunrise_value, tz_str),
            "sunset": _parse_local_datetime(sunset_value, tz_str),
        }
        for date_value, sunrise_value, sunset_value in zip(dates, sunrises, sunsets, strict=False)
    }

    today = now.date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    today_data = by_date.get(today)
    yesterday_data = by_date.get(yesterday)
    tomorrow_data = by_date.get(tomorrow)
    if not today_data or not yesterday_data or not tomorrow_data:
        return None

    return _build_prahara_context(
        now=now,
        sunrise_today=today_data["sunrise"],
        sunset_today=today_data["sunset"],
        sunrise_tomorrow=tomorrow_data["sunrise"],
        sunset_yesterday=yesterday_data["sunset"],
    )


async def fetch_weather_context(lat: float, lon: float, tz_str: str) -> WeatherContext:
    try:
        data = await _fetch_open_meteo_payload(lat, lon, tz_str)
        return _build_weather_context(data, tz_str)
    except Exception as exc:  # noqa: BLE001 - fallback is intentional for local dev.
        log.warning("Weather fetch failed, using local fallback: %s", exc)
        return _fallback_weather_context(tz_str)


async def get_environment_context(
    lat: float | None,
    lon: float | None,
    tz_str: str | None,
) -> tuple[PraharaContext, WeatherContext]:
    resolved_lat = lat if lat is not None else settings.default_lat
    resolved_lon = lon if lon is not None else settings.default_lon
    resolved_tz = tz_str or settings.default_tz
    try:
        data = await _fetch_open_meteo_payload(resolved_lat, resolved_lon, resolved_tz)
    except Exception as exc:  # noqa: BLE001 - fallback is intentional for local dev.
        log.warning("Environment fetch failed, using local fallback: %s", exc)
        return calculate_prahara(resolved_lat, resolved_lon, resolved_tz), _fallback_weather_context(
            resolved_tz
        )

    weather = _build_weather_context(data, resolved_tz)
    prahara = _build_live_prahara_context(data, resolved_tz)
    if prahara is None:
        prahara = calculate_prahara(resolved_lat, resolved_lon, resolved_tz)
    return prahara, weather


def serialize_prahara_context(context: PraharaContext) -> dict[str, Any]:
    data = asdict(context)
    data["sunrise"] = context.sunrise.isoformat()
    data["sunset"] = context.sunset.isoformat()
    return data


def serialize_weather_context(context: WeatherContext) -> dict[str, Any]:
    return asdict(context)
