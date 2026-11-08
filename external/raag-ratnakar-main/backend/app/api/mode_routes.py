from __future__ import annotations

from fastapi import APIRouter

from app.services.mode_engine import get_all_modes

router = APIRouter(prefix="/api", tags=["modes"])


@router.get("/modes")
def modes() -> dict[str, list[dict[str, object]]]:
    return {"modes": get_all_modes()}

