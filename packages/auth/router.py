"""Authentication router for RepoIntel.

Provides login, register, refresh, and current-user endpoints.
All endpoints are public (they are the authentication entry points).
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, EmailStr

from packages.auth.jwt_handler import (
    create_access_token,
    create_refresh_token,
    verify_token,
    require_auth,
)
from packages.auth.service import authenticate_user, create_user, get_user_by_id, user_to_pydantic
from packages.auth.models import User, UserRole, AuthResponse
from packages.knowledge.db import AsyncSessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """Register a new user. Public endpoint."""
    async with AsyncSessionLocal() as session:
        try:
            user_model = await create_user(
                session,
                username=request.username,
                email=request.email,
                password=request.password,
                role=UserRole.VIEWER,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            )

    user = await user_to_pydantic(user_model)
    return TokenResponse(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
        expires_in=30 * 60,
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate and receive access + refresh tokens. Public endpoint."""
    async with AsyncSessionLocal() as session:
        user_model = await authenticate_user(session, request.username, request.password)

    if user_model is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    user = await user_to_pydantic(user_model)
    return TokenResponse(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
        expires_in=30 * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: RefreshRequest):
    """Exchange a valid refresh token for a new access token."""
    payload = verify_token(request.refresh_token)
    async with AsyncSessionLocal() as session:
        user_model = await get_user_by_id(session, payload.sub)

    if user_model is None or not user_model.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    user = await user_to_pydantic(user_model)
    return TokenResponse(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
        expires_in=30 * 60,
    )


@router.get("/me", response_model=User)
async def get_me(current_user: User = Depends(require_auth())):
    """Get the current authenticated user."""
    return current_user


@router.get("/verify", response_model=User)
async def verify_access(current_user: User = Depends(require_auth())):
    """Verify that the current access token is valid."""
    return current_user