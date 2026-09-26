"""JWT token handling for RepoIntel.

Provides secure token creation, verification, and user resolution.
Uses HS256 signing with configurable secret and expiration.

Security:
    - Tokens are signed with a secret from environment (never hardcoded)
    - Short-lived access tokens (default 30 minutes)
    - Refresh tokens for obtaining new access tokens
    - Token ID (jti) for revocation support
"""

import os
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from packages.auth.models import User, UserRole, TokenPayload
from packages.auth.service import get_user_by_id, user_to_pydantic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_DEV_SECRET = None


def get_secret_key() -> str:
    """Resolve the current JWT signing secret from the live environment.

    The secret must be read at use time rather than at import time. RepoIntel
    initializes configuration during the app lifespan, and tests may set
    ``REPOINTEL_JWT_SECRET`` before or after importing the auth module. A
    module-level constant freezes the first value encountered and causes valid
    tokens to be rejected when the environment is populated later in the same
    process.
    """
    configured = os.getenv("REPOINTEL_JWT_SECRET")
    if configured:
        return configured

    global _DEFAULT_DEV_SECRET
    if _DEFAULT_DEV_SECRET is None:
        import secrets

        _DEFAULT_DEV_SECRET = secrets.token_hex(32)
        logger.warning(
            "REPOINTEL_JWT_SECRET not set; using ephemeral secret. "
            "Set this in production!"
        )
    return _DEFAULT_DEV_SECRET


SECRET_KEY = get_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("REPOINTEL_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REPOINTEL_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Bearer token scheme
bearer_scheme = HTTPBearer()


# ---------------------------------------------------------------------------
# Token creation
# ---------------------------------------------------------------------------

def create_access_token(user: User) -> str:
    """Create a JWT access token for a user."""
    now = datetime.now(timezone.utc)
    payload = TokenPayload(
        sub=user.user_id,
        username=user.username,
        role=user.role.value,
        exp=now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        iat=now,
        jti=str(uuid.uuid4()),
    )
    return jwt.encode(payload.model_dump(), get_secret_key(), algorithm=ALGORITHM)


def create_refresh_token(user: User) -> str:
    """Create a JWT refresh token for a user."""
    now = datetime.now(timezone.utc)
    payload = TokenPayload(
        sub=user.user_id,
        username=user.username,
        role=user.role.value,
        exp=now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        iat=now,
        jti=str(uuid.uuid4()),
    )
    return jwt.encode(payload.model_dump(), get_secret_key(), algorithm=ALGORITHM)


def create_auth_response(user: User) -> dict:
    """Create a full auth response with tokens."""
    return {
        "access_token": create_access_token(user),
        "refresh_token": create_refresh_token(user),
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": user.model_dump(),
    }


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------

def verify_token(token: str) -> TokenPayload:
    """Verify a JWT token and return its payload.
    
    Raises:
        HTTPException: If token is invalid, expired, or malformed.
    """
    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=[ALGORITHM])
        token_payload = TokenPayload(**payload)
        
        # Check expiration manually (should be handled by jose, but double-check)
        if token_payload.exp < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return token_payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Current user resolution
# ---------------------------------------------------------------------------

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> User:
    """Resolve the current user from a Bearer token.

    Extracts and validates the JWT, then performs a real database lookup
    to ensure the user still exists and is active. This prevents using
    revoked or deleted user tokens.

    Raises:
        HTTPException: If token is invalid or user not found.
    """
    token = credentials.credentials
    payload = verify_token(token)

    # Real database lookup — not just token reconstruction
    from packages.knowledge.db import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        user_model = await get_user_by_id(session, payload.sub)

    if user_model is None or not user_model.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await user_to_pydantic(user_model)


def require_auth(role: Optional[UserRole] = None):
    """Decorator/dependency factory for requiring authentication.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: User = Depends(require_auth())):
            ...
        
        @app.get("/admin-only")
        async def admin_route(user: User = Depends(require_auth(UserRole.ADMIN))):
            ...
    """
    async def _auth_dependency(
        credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    ) -> User:
        user = await get_current_user(credentials)
        
        if role and user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {role.value}",
            )
        
        return user
    
    return _auth_dependency