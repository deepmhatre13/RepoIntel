"""Generate docker-compose.yml with proper ${VAR:-default} syntax."""
import os

COMPOSE_YAML = """# RepoIntel production compose file.
#
# Deploys the PostgreSQL database, Redis broker, the Celery worker, the
# RepoIntel API/frontend container, and an nginx reverse proxy.
#
# Standard usage:
#     docker compose up -d
#
# Environment (set in your shell or a .env alongside this file):
#     GOOGLE_API_KEY        optional; when absent the API runs in deterministic
#                           degraded mode (Gemini unavailable is explicit, not fate).
#     POSTGRES_PASSWORD     set a strong value in production.

services:
  db:
    # PostgreSQL 16 with the pgvector extension pre-installed (maintained image).
    # The stock postgres:16-alpine image does NOT ship the `vector` extension,
    # which RepoIntel's knowledge layer requires (pgvector.sqlalchemy.Vector).
    image: pgvector/pgvector:pg16
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-repointel}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-repointel_local_dev_password}
      POSTGRES_DB: ${POSTGRES_DB:-repointel}
    volumes:
      - repointel_pgdata:/var/lib/postgresql/data
      # Enable the vector extension on first database initialization (idempotent).
      - ./docker/db/initdb:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-repointel} -d ${POSTGRES_DB:-repointel}"]
      interval: 5s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy noeviction
    volumes:
      - repointel_redis:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: backend
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-repointel}:${POSTGRES_PASSWORD:-repointel_local_dev_password}@db:5432/${POSTGRES_DB:-repointel}
      GOOGLE_API_KEY: ${GOOGLE_API_KEY:-}
      GEMINI_MODEL: ${GEMINI_MODEL:-gemini-2.5-flash}
      EMBEDDING_PROVIDER: ${EMBEDDING_PROVIDER:-local}
      EMBEDDING_MODEL: ${EMBEDDING_MODEL:-all-MiniLM-L6-v2}
      EMBEDDING_BATCH_SIZE: ${EMBEDDING_BATCH_SIZE:-32}
      REPOINTEL_CORS_ORIGINS: ${REPOINTEL_CORS_ORIGINS:-}
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
      REPOINTEL_WORKERS: ${REPOINTEL_WORKERS:-2}
      MAX_REPOSITORY_FILES: ${MAX_REPOSITORY_FILES:-20000}
      MAX_FILE_SIZE: ${MAX_FILE_SIZE:-2097152}
      MAX_LLM_CALLS: ${MAX_LLM_CALLS:-25}
      MAX_EXECUTION_SECONDS: ${MAX_EXECUTION_SECONDS:-900}
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/1
    # Direct developer access to the API (http://localhost:8000) is intentionally
    # published: the README and test-suite interact with /health, /llm-status and
    # /openapi.json directly on this port. The frontend also reaches the API here.
    ports:
      - "${REPOINTEL_API_PORT:-8000}:8000"

  worker:
    build:
      context: .
      dockerfile: Dockerfile
      target: backend
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
      # Migrations are owned by the API container (docker-entrypoint.sh runs
      # `alembic upgrade head` before serving). Start the worker only once the
      # API is healthy so it never races schema setup.
      api:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-repointel}:${POSTGRES_PASSWORD:-repointel_local_dev_password}@db:5432/${POSTGRES_DB:-repointel}
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/1
      WORKER_CONCURRENCY: ${WORKER_CONCURRENCY:-2}
      GOOGLE_API_KEY: ${GOOGLE_API_KEY:-}
      GEMINI_MODEL: ${GEMINI_MODEL:-gemini-2.5-flash}
      EMBEDDING_PROVIDER: ${EMBEDDING_PROVIDER:-local}
      EMBEDDING_MODEL: ${EMBEDDING_MODEL:-all-MiniLM-L6-v2}
      EMBEDDING_BATCH_SIZE: ${EMBEDDING_BATCH_SIZE:-32}
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
      MAX_REPOSITORY_FILES: ${MAX_REPOSITORY_FILES:-20000}
      MAX_FILE_SIZE: ${MAX_FILE_SIZE:-2097152}
      MAX_LLM_CALLS: ${MAX_LLM_CALLS:-25}
      MAX_EXECUTION_SECONDS: ${MAX_EXECUTION_SECONDS:-900}
    command: python -m packages.worker.start_worker

  frontend:
    build:
      context: .
      dockerfile: Dockerfile
      target: frontend
    restart: unless-stopped
    depends_on:
      - api
    ports:
      - "${REPOINTEL_HTTP_PORT:-8080}:80"

volumes:
  repointel_pgdata:
  repointel_redis:

networks:
  default:
    driver: bridge
"""


def main():
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docker-compose.yml")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(COMPOSE_YAML)
    print(f"Generated {output_path} ({len(COMPOSE_YAML)} bytes)")


if __name__ == "__main__":
    main()
