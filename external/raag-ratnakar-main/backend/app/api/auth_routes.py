from __future__ import annotations

from fastapi import APIRouter

from app.schemas.auth import AuthRequest, AuthResponse
from app.services.auth_service import authenticate_user, create_access_token, register_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(payload: AuthRequest) -> AuthResponse:
    user = register_user(payload.email, payload.password)
    return AuthResponse(access_token=create_access_token(user), user=user)


@router.post("/login", response_model=AuthResponse)
def login(payload: AuthRequest) -> AuthResponse:
    user = authenticate_user(payload.email, payload.password)
    return AuthResponse(access_token=create_access_token(user), user=user)

