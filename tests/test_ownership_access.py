"""Runtime ownership/access security tests.

Tests the user→repository ownership model through actual HTTP requests:
- User A creates and can access own repository
- User B cannot access User A's repository (404, not leaking existence)
- Unauthenticated requests are rejected (401)
- run/artifact access follows ownership
"""
import uuid

import pytest
from fastapi.testclient import TestClient

from apps.api import app


def _register_user(client, prefix: str) -> dict:
    uname = f"{prefix}_{uuid.uuid4().hex[:8]}"
    res = client.post("/auth/register", json={
        "username": uname,
        "email": f"{uname}@test.com",
        "password": "testpass123",
    })
    assert res.status_code == 201, f"Register failed: {res.text}"
    token = res.json()["access_token"]
    return {
        "headers": {"Authorization": f"Bearer {token}"},
        "token": token,
        "username": uname,
    }


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def user_a(client):
    return _register_user(client, "test_owner_a")


@pytest.fixture()
def user_b(client):
    return _register_user(client, "test_owner_b")


def test_repository_creation_assigns_ownership(client, user_a):
    res = client.post(
        "/repositories",
        json={"repo_url": "https://github.com/example/owner-test"},
        headers=user_a["headers"],
    )
    assert res.status_code == 200
    assert "repository_id" in res.json()


def test_repeated_repository_creation_is_idempotent(client, user_a):
    repo_url = "https://github.com/example/repeated-owner"
    first = client.post("/repositories", json={"repo_url": repo_url}, headers=user_a["headers"])
    second = client.post("/repositories", json={"repo_url": repo_url}, headers=user_a["headers"])

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["repository_id"] == second.json()["repository_id"]


def test_owner_can_read_dependencies(client, user_a):
    res = client.post(
        "/repositories",
        json={"repo_url": "https://github.com/example/owner-read"},
        headers=user_a["headers"],
    )
    repo_id = res.json()["repository_id"]

    res = client.get(f"/repositories/{repo_id}/dependencies", headers=user_a["headers"])
    assert res.status_code == 200
    assert "dependencies" in res.json()


def test_other_user_cannot_access_owner_repo(client, user_a, user_b):
    res = client.post(
        "/repositories",
        json={"repo_url": "https://github.com/example/owner-priv"},
        headers=user_a["headers"],
    )
    repo_id = res.json()["repository_id"]

    res = client.get(f"/repositories/{repo_id}/dependencies", headers=user_b["headers"])
    assert res.status_code == 404
    assert "detail" in res.json()


@pytest.mark.integration
def test_owner_can_analyze(client, user_a):
    res = client.post(
        "/repositories",
        json={"repo_url": "https://github.com/example/owner-analyze"},
        headers=user_a["headers"],
    )
    repo_id = res.json()["repository_id"]

    res = client.post(
        f"/repositories/{repo_id}/analyze",
        json={"repo_url": "https://github.com/example/owner-analyze"},
        headers=user_a["headers"],
    )
    assert res.status_code == 200
    assert "run_id" in res.json()


@pytest.mark.integration
def test_other_user_cannot_analyze(client, user_a, user_b):
    res = client.post(
        "/repositories",
        json={"repo_url": "https://github.com/example/owner-analyze2"},
        headers=user_a["headers"],
    )
    repo_id = res.json()["repository_id"]

    res = client.post(
        f"/repositories/{repo_id}/analyze",
        json={"repo_url": "https://github.com/example/owner-analyze2"},
        headers=user_b["headers"],
    )
    assert res.status_code == 404


def test_unauthenticated_rejected(client):
    res = client.get("/repositories/does-not-exist/dependencies")
    assert res.status_code == 401

    res = client.get(
        "/repositories/does-not-exist/dependencies",
        headers={"Authorization": "Bearer invalid-token-here"},
    )
    assert res.status_code in (401, 422)