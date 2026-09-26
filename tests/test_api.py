import pytest
from fastapi.testclient import TestClient
from apps.api import app


@pytest.fixture(scope="module")
def client():
    """Context-managed TestClient so the FastAPI lifespan runs, which invokes
    init_db() migrations. Without this, an un-migrated (fresh) database lacks
    the tables queried by the read-only endpoints."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth_headers(client):
    """Register a user and return auth headers for protected endpoints."""
    # Register a test user
    import uuid
    uname = f"testuser_{uuid.uuid4().hex[:8]}"
    res = client.post("/auth/register", json={
        "username": uname,
        "email": f"{uname}@test.com",
        "password": "testpass123",
    })
    assert res.status_code == 201, f"Register failed: {res.text}"
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_api_endpoints(client, auth_headers):
    # 1. Create repository (requires auth)
    res1 = client.post("/repositories", json={"repo_url": "https://github.com/example/demo"}, headers=auth_headers)
    assert res1.status_code == 200
    repo_id = res1.json()["repository_id"]

    # 2. Dependencies read-only endpoint (requires auth)
    res2 = client.get(f"/repositories/{repo_id}/dependencies", headers=auth_headers)
    assert res2.status_code == 200
    assert "dependencies" in res2.json()

    # 3. Graph read-only endpoint (requires auth)
    res3 = client.get(f"/repositories/{repo_id}/graph", headers=auth_headers)
    assert res3.status_code == 200
    assert "neighbors" in res3.json()


def test_api_llm_status_and_health(client):
    # /health is a simple liveness probe.
    assert client.get("/health").status_code == 200
    # /llm-status reports a live/ degraded mode explicitly and never fakes it.
    ls = client.get("/llm-status").json()
    assert ls["mode"] in ("LIVE_LLM", "DEGRADED_NO_LLM")
    assert ls["llm_available"] in (True, False)
    assert ls["model"] == "gemini-3.8-flash"


def test_api_unknown_repo_404(client, auth_headers):
    # Unknown repository returns structured JSON 404, not a raw traceback.
    res = client.get("/repositories/does-not-exist/dependencies", headers=auth_headers)
    assert res.status_code == 404
    assert "detail" in res.json()


def test_api_unauthenticated_401(client):
    # Protected endpoints must reject unauthenticated requests.
    res = client.post("/repositories", json={"repo_url": "https://github.com/example/demo"})
    assert res.status_code == 401
    res = client.get("/repositories/anything/dependencies")
    assert res.status_code == 401


def test_dispatch_error_classification(client, auth_headers, monkeypatch):
    from unittest.mock import AsyncMock, patch
    from packages.worker.dispatch import DispatchError

    # Create repository record with auth ownership
    repo_res = client.post("/repositories", json={"repo_url": "https://github.com/octocat/Hello-World"}, headers=auth_headers)
    assert repo_res.status_code == 200
    repo_id = repo_res.json()["repository_id"]

    # 1. Broker unreachable exception -> 503
    with patch("apps.job_runner.JobRunner.dispatch_durable", side_effect=DispatchError("Celery broker 'redis://redis:6379/0' is unreachable: Connection refused")):
        res = client.post(
            f"/repositories/{repo_id}/analyze",
            json={"repo_url": "https://github.com/octocat/Hello-World"},
            headers=auth_headers,
        )
        assert res.status_code == 503
        assert "broker is unreachable" in res.json()["detail"]

    # 2. Non-broker exception -> 500 with exact error details (not masked as broker unreachable)
    with patch("apps.job_runner.JobRunner.dispatch_durable", side_effect=ValueError("Invalid task serialization state")):
        res = client.post(
            f"/repositories/{repo_id}/analyze",
            json={"repo_url": "https://github.com/octocat/Hello-World"},
            headers=auth_headers,
        )
        assert res.status_code == 500
        detail = res.json()["detail"]
        assert "ValueError" in detail
        assert "Invalid task serialization state" in detail
        assert "broker is unreachable" not in detail
