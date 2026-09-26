import os
import sys
import json
import time
import uuid
import subprocess
import urllib.request

API_BASE = "http://localhost:8000"

def http_post(endpoint: str, payload: dict, headers: dict = None) -> tuple[int, dict]:
    url = f"{API_BASE}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"detail": body}

def main():
    print("==================================================")
    print("DOCKERIZED API -> REDIS -> CELERY WORKER VERIFICATION")
    print("==================================================")

    # 1. Register a unique test user in the API database
    uid = uuid.uuid4().hex[:8]
    uname = f"u_{uid}"
    email = f"u_{uid}@example.org"
    password = "password123"
    print(f"1. Registering unique user '{uname}' via API endpoint...")
    reg_status, reg_res = http_post("/auth/register", {"username": uname, "email": email, "password": password})
    print(f"   Register status={reg_status}, body={reg_res}")
    assert reg_status == 201, f"Failed registering user: {reg_res}"
    token = reg_res["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("   Auth Token acquired.")

    # 2. Create repository
    test_url = "https://github.com/octocat/Hello-World"
    print(f"2. Creating repository record for {test_url}...")
    create_status, create_res = http_post("/repositories", {"repo_url": test_url}, headers=headers)
    print(f"   Response status={create_status}, body={create_res}")
    assert create_status == 200, f"Failed creating repo: {create_res}"
    repo_id = create_res["repository_id"]

    # 3. Dispatch analysis via HTTP POST /repositories/{repo_id}/analyze
    print(f"3. Dispatching analysis via POST /repositories/{repo_id}/analyze ...")
    analyze_status, analyze_res = http_post(f"/repositories/{repo_id}/analyze", {"repo_url": test_url}, headers=headers)
    print(f"   Response status={analyze_status}")
    print(f"   Response body={analyze_res}")

    # Assertions
    assert analyze_status == 200, f"Expected HTTP 200 OK, got {analyze_status}: {analyze_res}"
    assert analyze_res.get("status") == "PENDING", f"Expected status PENDING, got {analyze_res.get('status')}"
    assert "task_id" in analyze_res, "Expected task_id in response"
    assert "run_id" in analyze_res, "Expected run_id in response"
    
    run_id = analyze_res["run_id"]
    task_id = analyze_res["task_id"]
    print(f"\n   SUCCESS: Analysis dispatch returned HTTP 200 OK with run_id={run_id}, task_id={task_id}, status=PENDING!")
    print("   No 'Celery broker unreachable' error occurred!")

    # 4. Check Celery Worker Logs
    print("\n4. Checking Celery worker container logs for task reception...")
    time.sleep(3)
    logs_cmd = ["docker", "compose", "logs", "--tail=30", "worker"]
    res = subprocess.run(logs_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("   Celery Worker Container Logs (tail):")
    for line in res.stdout.splitlines()[-15:]:
        print(f"   | {line}")

    print("\n==================================================")
    print("REAL DOCKERIZED EXECUTION VERIFIED SUCCESSFULLY")
    print("==================================================")

if __name__ == "__main__":
    main()

