from fastapi.testclient import TestClient
from apps.api import app

client = TestClient(app)

def test_api_endpoints():
    # 1. Create repository
    res1 = client.post("/repositories", json={"repo_url": "https://github.com/example/demo"})
    assert res1.status_code == 200
    repo_id = res1.json()["repository_id"]

    # 2. Dependencies read-only endpoint
    res2 = client.get(f"/repositories/{repo_id}/dependencies")
    assert res2.status_code == 200
    assert "dependencies" in res2.json()

    # 3. Graph read-only endpoint
    res3 = client.get(f"/repositories/{repo_id}/graph")
    assert res3.status_code == 200
    assert "neighbors" in res3.json()
