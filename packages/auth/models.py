"""Authentication models for RepoIntel."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """User roles for authorization."""
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


class User(BaseModel):
    """Authenticated user model."""
    user_id: str
    username: str
    email: str
    role: UserRole = UserRole.VIEWER
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # user_id
    username: str
    role: str
    exp: datetime  # expiration
    iat: datetime  # issued at
    jti: str  # unique token ID


class AuthResponse(BaseModel):
    """Authentication response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: User


class TokenRefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class LoginRequest(BaseModel):
    """Login request."""
    username: str
    password: str
