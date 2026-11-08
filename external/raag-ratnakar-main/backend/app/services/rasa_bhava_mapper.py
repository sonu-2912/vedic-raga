from __future__ import annotations


RASA_TO_BHAVA: dict[str, str] = {
    "shringara": "rati",
    "hasya": "hasa",
    "karuna": "shoka",
    "raudra": "krodha",
    "veer": "utsaha",
    "bhayanak": "bhaya",
    "adhbuta": "vismaya",
    "shanta": "shama",
    # The current raga dataset has no Jugupsa entries, so local v1 falls back
    # to Shama to keep recommendations usable and calming.
    "bhibhatsa": "shama",
}


def map_rasa_to_bhava(rasa: str) -> str:
    normalized = rasa.lower().strip()
    if normalized not in RASA_TO_BHAVA:
        raise ValueError(f"Unsupported rasa '{rasa}'")
    return RASA_TO_BHAVA[normalized]

