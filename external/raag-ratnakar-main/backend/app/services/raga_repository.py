from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import settings


@dataclass(frozen=True)
class Raga:
    raga_name: str
    primary_sthayi_bhava: str
    secondary_sthayi_bhava: list[str]
    prahara: list[int]
    ritu: str
    youtube_links: dict[str, list[str]]


class RagaRepository:
    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = data_path or settings.raga_data_path
        self._ragas: list[Raga] | None = None
        self._by_name: dict[str, Raga] | None = None

    def _load(self) -> None:
        if self._ragas is not None:
            return
        if not self.data_path.exists():
            raise FileNotFoundError(f"Raga dataset not found: {self.data_path}")

        raw = json.loads(self.data_path.read_text(encoding="utf-8"))
        ragas: list[Raga] = []
        for item in raw:
            prahara = item.get("prahara", [])
            if isinstance(prahara, int):
                prahara = [prahara]
            ragas.append(
                Raga(
                    raga_name=str(item.get("raga_name", "")).strip(),
                    primary_sthayi_bhava=str(item.get("primary_sthayi_bhava", "")).strip(),
                    secondary_sthayi_bhava=[
                        str(value).strip() for value in item.get("secondary_sthayi_bhava", [])
                    ],
                    prahara=[int(value) for value in prahara if value],
                    ritu=str(item.get("ritu", "")).strip().lower(),
                    youtube_links={
                        "vocal": list(item.get("youtube_links", {}).get("vocal", [])),
                        "instrumental": list(item.get("youtube_links", {}).get("instrumental", [])),
                    },
                )
            )
        self._ragas = ragas
        self._by_name = {raga.raga_name.lower(): raga for raga in ragas}

    def all(self) -> list[Raga]:
        self._load()
        return list(self._ragas or [])

    def get_by_name(self, name: str) -> Raga | None:
        self._load()
        return (self._by_name or {}).get(name.lower().strip())

