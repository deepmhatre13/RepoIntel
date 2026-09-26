"""Authentication tests for RepoIntel.

Tests JWT token creation, verification, and password handling.
"""

import os
import pytest
from datetime import datetime, timedelta, timezone

# Set a test secret before importing auth modules
os.environ["REPOINTEL_JWT_SECRET"] = "test-secret-key-for-unit-tests-only"

from packages.auth.jwt_handler import (
    create_access_token,
    create_refresh_token,
    verify_token,
    create_auth_response,
    SECRET_KEY,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from packages.auth.service import hash_password, verify_password
from packages.auth.models import User, UserRole, TokenPayload


class TestTokenCreation:
    """Test JWT token creation."""

    def test_create_access_token(self):
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            role=UserRole.VIEWER,
        )
        token = create_access_token(user)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self):
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            role=UserRole.VIEWER,
        )
        token = create_refresh_token(user)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_access_and_refresh_tokens_differ(self):
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            role=UserRole.VIEWER,
        )
        access = create_access_token(user)
        refresh = create_refresh_token(user)
        assert access != refresh

    def test_create_auth_response(self):
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            role=UserRole.ANALYST,
        )
        response = create_auth_response(user)
        assert "access_token" in response
        assert "refresh_token" in response
        assert response["token_type"] == "bearer"
        assert response["expires_in"] == ACCESS_TOKEN_EXPIRE_MINUTES * 60
        assert response["user"]["username"] == "testuser"


class TestTokenVerification:
    """Test JWT token verification."""

    def test_verify_valid_token(self):
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            role=UserRole.ADMIN,
        )
        token = create_access_token(user)
        payload = verify_token(token)
        
        assert payload.sub == "user1"
        assert payload.username == "testuser"
        assert payload.role == "admin"
        assert payload.exp > datetime.now(timezone.utc)

    def test_verify_invalid_token(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            verify_token("invalid.token.here")
        assert exc_info.value.status_code == 401

    def test_verify_tampered_token(self):
        from fastapi import HTTPException
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
        )
        token = create_access_token(user)
        # Tamper with the token
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(HTTPException) as exc_info:
            verify_token(tampered)
        assert exc_info.value.status_code == 401

    def test_verify_wrong_secret(self):
        from fastapi import HTTPException
        import jwt as _jwt
        # Create a token with a different secret
        payload = {
            "sub": "user1",
            "username": "testuser",
            "role": "viewer",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=30),
            "iat": datetime.now(timezone.utc),
            "jti": "test-jti",
        }
        wrong_token = _jwt.encode(payload, "wrong-secret", algorithm="HS256")
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(wrong_token)
        assert exc_info.value.status_code == 401


class TestPasswordHandling:
    """Test password hashing and verification."""

    def test_hash_password(self):
        hashed = hash_password("testpassword123")
        assert isinstance(hashed, str)
        assert hashed != "testpassword123"
        assert hashed.startswith("$2b$")  # bcrypt prefix

    def test_verify_correct_password(self):
        password = "testpassword123"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_verify_incorrect_password(self):
        password = "testpassword123"
        hashed = hash_password(password)
        assert verify_password("wrongpassword", hashed) is False

    def test_different_passwords_different_hashes(self):
        password = "testpassword123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        # bcrypt generates different salts, so hashes should differ
        assert hash1 != hash2
        # But both should verify correctly
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True


class TestUserModel:
    """Test user model."""

    def test_create_user(self):
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
        )
        assert user.user_id == "user1"
        assert user.role == UserRole.VIEWER  # default
        assert user.is_active is True

    def test_create_admin_user(self):
        user = User(
            user_id="admin1",
            username="admin",
            email="admin@example.com",
            role=UserRole.ADMIN,
        )
        assert user.role == UserRole.ADMIN
