from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

VALID_BHAVAS = frozenset(
    {
        "rati",
        "hasa",
        "shoka",
        "krodha",
        "utsaha",
        "bhaya",
        "vismaya",
        "shama",
    }
)


class Mode(str, Enum):
    THERAPEUTIC = "therapeutic"
    TRADITIONAL = "traditional"
    STUDY = "study"
    GYM = "gym"
    SLEEP = "sleep"
    MEDITATION = "meditation"


_THERAPEUTIC_MAP: dict[str, list[tuple[str, float]]] = {
    "krodha": [("utsaha", 0.50), ("shama", 0.50)],
    "bhaya": [("vismaya", 0.50), ("shama", 0.50)],
    "shoka": [("shama", 0.50), ("rati", 0.50)],
    "rati": [("rati", 1.00)],
    "hasa": [("hasa", 1.00)],
    "utsaha": [("utsaha", 1.00)],
    "vismaya": [("vismaya", 1.00)],
    "shama": [("shama", 1.00)],
}

_TRADITIONAL_MAP: dict[str, list[tuple[str, float]]] = {
    "krodha": [("krodha", 0.30), ("utsaha", 0.40), ("shama", 0.30)],
    "bhaya": [("bhaya", 0.30), ("vismaya", 0.40), ("shama", 0.30)],
    "shoka": [("shoka", 0.40), ("shama", 0.60)],
    "rati": [("rati", 0.70), ("shama", 0.30)],
    "hasa": [("hasa", 0.70), ("shama", 0.30)],
    "utsaha": [("utsaha", 0.70), ("shama", 0.30)],
    "vismaya": [("vismaya", 0.70), ("shama", 0.30)],
    "shama": [("shama", 1.00)],
}

_EMOTION_AWARE_MAPS: dict[str, dict[str, list[tuple[str, float]]]] = {
    Mode.THERAPEUTIC.value: _THERAPEUTIC_MAP,
    Mode.TRADITIONAL.value: _TRADITIONAL_MAP,
}

_ACTIVITY_RAGAS: dict[str, list[str]] = {
    Mode.STUDY.value: ["Bhoopali", "Yaman", "Darbari Kanada"],
    Mode.GYM.value: ["Miyan Ki Todi", "Pooriya", "Shanmukhapriya"],
    Mode.SLEEP.value: ["Darbari Kanada", "Bageshree", "Bhimpalasi"],
    Mode.MEDITATION.value: ["Bhairavi", "Bhoopali", "Durga"],
}


@dataclass(frozen=True)
class EmotionAwareResult:
    mode: str
    detected_bhava: str
    targets: list[tuple[str, float]]


@dataclass(frozen=True)
class ActivityResult:
    mode: str
    raga_names: list[str]


def is_activity_mode(mode: str) -> bool:
    return mode.lower().strip() in _ACTIVITY_RAGAS


def is_emotion_aware_mode(mode: str) -> bool:
    return mode.lower().strip() in _EMOTION_AWARE_MAPS


def get_recommendation(mode: str, detected_bhava: str | None = None) -> EmotionAwareResult | ActivityResult:
    normalized_mode = mode.lower().strip()

    if normalized_mode in _ACTIVITY_RAGAS:
        return ActivityResult(mode=normalized_mode, raga_names=list(_ACTIVITY_RAGAS[normalized_mode]))

    if normalized_mode not in _EMOTION_AWARE_MAPS:
        valid = [m.value for m in Mode]
        raise ValueError(f"Unknown mode '{mode}'. Valid modes: {valid}")

    if detected_bhava is None:
        raise ValueError(f"Mode '{normalized_mode}' requires detected_bhava.")

    normalized_bhava = detected_bhava.lower().strip()
    if normalized_bhava not in VALID_BHAVAS:
        raise ValueError(f"Unknown bhava '{detected_bhava}'. Valid bhavas: {sorted(VALID_BHAVAS)}")

    return EmotionAwareResult(
        mode=normalized_mode,
        detected_bhava=normalized_bhava,
        targets=list(_EMOTION_AWARE_MAPS[normalized_mode][normalized_bhava]),
    )


def get_all_modes() -> list[dict[str, object]]:
    return [
        {
            "id": Mode.THERAPEUTIC.value,
            "name": "Therapeutic",
            "description": "Guides the recommendation toward steadier emotional balance.",
            "requires_image": True,
            "default": True,
        },
        {
            "id": Mode.TRADITIONAL.value,
            "name": "Traditional",
            "description": "Follows a classical rasa journey while preserving the detected mood.",
            "requires_image": True,
            "default": False,
        },
        {
            "id": Mode.STUDY.value,
            "name": "Study",
            "description": "Curated ragas for focus and sustained attention.",
            "requires_image": False,
            "default": False,
        },
        {
            "id": Mode.GYM.value,
            "name": "Gym",
            "description": "Rhythmic and energetic ragas for movement.",
            "requires_image": False,
            "default": False,
        },
        {
            "id": Mode.SLEEP.value,
            "name": "Sleep",
            "description": "Late, calm ragas for rest and slowing down.",
            "requires_image": False,
            "default": False,
        },
        {
            "id": Mode.MEDITATION.value,
            "name": "Meditation",
            "description": "Gentle ragas for stillness and internal quiet.",
            "requires_image": False,
            "default": False,
        },
    ]

