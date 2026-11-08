from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.mode_engine import ActivityResult, EmotionAwareResult, get_recommendation
from app.services.raga_repository import Raga, RagaRepository


@dataclass(frozen=True)
class RecommendationResult:
    raga_name: str
    score: float
    target_bhava: str
    bhava_match_type: str
    bhava_score: float
    prahara_score: float
    ritu_score: float
    primary_sthayi_bhava: str
    secondary_sthayi_bhava: list[str]
    prahara: list[int]
    ritu: str
    youtube_links: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raga_name": self.raga_name,
            "score": self.score,
            "target_bhava": self.target_bhava,
            "bhava_match_type": self.bhava_match_type,
            "bhava_score": self.bhava_score,
            "prahara_score": self.prahara_score,
            "ritu_score": self.ritu_score,
            "primary_sthayi_bhava": self.primary_sthayi_bhava,
            "secondary_sthayi_bhava": self.secondary_sthayi_bhava,
            "prahara": self.prahara,
            "ritu": self.ritu,
            "youtube_links": self.youtube_links,
        }


THERAPEUTIC_RAGA_LIMIT = 2
THERAPEUTIC_SONG_LIMIT = 4
TRADITIONAL_RAGA_LIMIT = 3
TRADITIONAL_SONG_LIMIT = 2


def _filter_links(links: dict[str, list[str]], style: str, limit: int | None = None) -> dict[str, list[str]]:
    normalized = style.lower().strip()
    vocal = list(links.get("vocal", []))
    instrumental = list(links.get("instrumental", []))

    if normalized == "vocal":
        selected_vocal = vocal[:limit] if limit is not None else vocal
        return {"vocal": selected_vocal, "instrumental": []}
    if normalized == "instrumental":
        selected_instrumental = instrumental[:limit] if limit is not None else instrumental
        return {"vocal": [], "instrumental": selected_instrumental}

    if limit is None:
        return {
            "vocal": vocal,
            "instrumental": instrumental,
        }

    flat_links: list[tuple[str, str]] = [("vocal", link) for link in vocal] + [
        ("instrumental", link) for link in instrumental
    ]
    limited = flat_links[:limit]
    selected = {"vocal": [], "instrumental": []}
    for kind, link in limited:
        selected[kind].append(link)
    return selected


def _score_raga(
    *,
    raga: Raga,
    target_bhava: str,
    target_weight: float,
    current_prahara: int,
    current_ritu: str,
    style: str,
    song_limit: int | None,
) -> RecommendationResult | None:
    primary = raga.primary_sthayi_bhava.lower()
    secondary = [value.lower() for value in raga.secondary_sthayi_bhava]
    target = target_bhava.lower()

    if target == primary:
        bhava_score = 1.0
        match_type = "primary"
    elif target in secondary:
        bhava_score = 0.6
        match_type = "secondary"
    else:
        return None

    prahara_score = 1.0 if current_prahara in raga.prahara else 0.0
    if not raga.ritu:
        ritu_score = 0.5
    else:
        ritu_score = 1.0 if raga.ritu == current_ritu.lower() else 0.0

    base_score = (bhava_score * 0.65) + (prahara_score * 0.20) + (ritu_score * 0.15)
    final_score = round(base_score * target_weight, 6)

    return RecommendationResult(
        raga_name=raga.raga_name,
        score=final_score,
        target_bhava=target,
        bhava_match_type=match_type,
        bhava_score=bhava_score,
        prahara_score=prahara_score,
        ritu_score=ritu_score,
        primary_sthayi_bhava=raga.primary_sthayi_bhava,
        secondary_sthayi_bhava=raga.secondary_sthayi_bhava,
        prahara=raga.prahara,
        ritu=raga.ritu,
        youtube_links=_filter_links(raga.youtube_links, style, song_limit),
    )


def _match_priority(match_type: str) -> int:
    if match_type == "primary":
        return 2
    if match_type == "secondary":
        return 1
    return 0


def _selection_limits(mode: str) -> tuple[int | None, int | None]:
    normalized = mode.lower().strip()
    if normalized == "therapeutic":
        return THERAPEUTIC_RAGA_LIMIT, THERAPEUTIC_SONG_LIMIT
    if normalized == "traditional":
        return TRADITIONAL_RAGA_LIMIT, TRADITIONAL_SONG_LIMIT
    return None, None


class RagaRecommender:
    def __init__(self, repository: RagaRepository) -> None:
        self.repository = repository

    def recommend(
        self,
        *,
        mode: str,
        detected_bhava: str | None,
        current_prahara: int,
        current_ritu: str,
        style: str = "both",
    ) -> list[RecommendationResult]:
        mode_result = get_recommendation(mode, detected_bhava)
        _, song_limit = _selection_limits(mode_result.mode)
        if isinstance(mode_result, ActivityResult):
            return self._recommend_activity(mode_result, current_prahara, current_ritu, style, song_limit)
        return self._recommend_emotion(mode_result, current_prahara, current_ritu, style, song_limit)

    def _recommend_activity(
        self,
        mode_result: ActivityResult,
        current_prahara: int,
        current_ritu: str,
        style: str,
        song_limit: int | None,
    ) -> list[RecommendationResult]:
        results: list[RecommendationResult] = []
        for index, name in enumerate(mode_result.raga_names):
            raga = self.repository.get_by_name(name)
            if not raga:
                continue
            prahara_score = 1.0 if current_prahara in raga.prahara else 0.0
            ritu_score = 1.0 if raga.ritu == current_ritu.lower() else 0.0
            results.append(
                RecommendationResult(
                    raga_name=raga.raga_name,
                    score=round(1.0 - (index * 0.01), 6),
                    target_bhava=mode_result.mode,
                    bhava_match_type="curated",
                    bhava_score=1.0,
                    prahara_score=prahara_score,
                    ritu_score=ritu_score,
                    primary_sthayi_bhava=raga.primary_sthayi_bhava,
                    secondary_sthayi_bhava=raga.secondary_sthayi_bhava,
                    prahara=raga.prahara,
                    ritu=raga.ritu,
                    youtube_links=_filter_links(raga.youtube_links, style, song_limit),
                )
            )
        return results

    def _recommend_emotion(
        self,
        mode_result: EmotionAwareResult,
        current_prahara: int,
        current_ritu: str,
        style: str,
        song_limit: int | None,
    ) -> list[RecommendationResult]:
        if mode_result.mode == "therapeutic":
            return self._recommend_therapeutic(mode_result, current_prahara, current_ritu, style, song_limit)
        return self._recommend_traditional(mode_result, current_prahara, current_ritu, style, song_limit)

    def _recommend_therapeutic(
        self,
        mode_result: EmotionAwareResult,
        current_prahara: int,
        current_ritu: str,
        style: str,
        song_limit: int | None,
    ) -> list[RecommendationResult]:
        best_by_name: dict[str, RecommendationResult] = {}
        for target_bhava, target_weight in mode_result.targets:
            for raga in self.repository.all():
                candidate = _score_raga(
                    raga=raga,
                    target_bhava=target_bhava,
                    target_weight=target_weight,
                    current_prahara=current_prahara,
                    current_ritu=current_ritu,
                    style=style,
                    song_limit=song_limit,
                )
                if candidate is None:
                    continue

                existing = best_by_name.get(candidate.raga_name)
                if existing is None:
                    best_by_name[candidate.raga_name] = candidate
                    continue

                candidate_key = (_match_priority(candidate.bhava_match_type), candidate.score)
                existing_key = (_match_priority(existing.bhava_match_type), existing.score)
                if candidate_key > existing_key:
                    best_by_name[candidate.raga_name] = candidate

        ranked = sorted(
            best_by_name.values(),
            key=lambda item: (
                -_match_priority(item.bhava_match_type),
                -item.score,
                item.raga_name.lower(),
            ),
        )
        return ranked[:THERAPEUTIC_RAGA_LIMIT]

    def _recommend_traditional(
        self,
        mode_result: EmotionAwareResult,
        current_prahara: int,
        current_ritu: str,
        style: str,
        song_limit: int | None,
    ) -> list[RecommendationResult]:
        selected: list[RecommendationResult] = []
        used_ragas: set[str] = set()

        for target_bhava, target_weight in mode_result.targets:
            ranked_for_target: list[RecommendationResult] = []
            for raga in self.repository.all():
                candidate = _score_raga(
                    raga=raga,
                    target_bhava=target_bhava,
                    target_weight=target_weight,
                    current_prahara=current_prahara,
                    current_ritu=current_ritu,
                    style=style,
                    song_limit=song_limit,
                )
                if candidate is not None:
                    ranked_for_target.append(candidate)

            ranked_for_target.sort(
                key=lambda item: (
                    -_match_priority(item.bhava_match_type),
                    -item.score,
                    item.raga_name.lower(),
                )
            )

            for candidate in ranked_for_target:
                if candidate.raga_name not in used_ragas:
                    selected.append(candidate)
                    used_ragas.add(candidate.raga_name)
                    break

        selected.sort(
            key=lambda item: (
                -_match_priority(item.bhava_match_type),
                -item.score,
                item.raga_name.lower(),
            )
        )
        return selected[:TRADITIONAL_RAGA_LIMIT]
