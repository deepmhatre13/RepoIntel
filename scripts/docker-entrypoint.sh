#!/bin/sh
# RepoIntel production entrypoint.
#
# 1. Run Alembic migrations (authoritative schema).
# 2. Start the FastAPI server with uvicorn.

set -e

echo "[entrypoint] running database migrations..."
python -m alembic upgrade head

echo "[entrypoint] starting RepoIntel API on 0.0.0.0:8000"
exec uvicorn apps.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "${REPOINTEL_WORKERS:-2}" \
    --proxy-headers \
    --no-access-log