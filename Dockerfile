# Dockerfile — RepoIntel production build
#
# Three stages:
#   1. Node build of the Vite frontend -> dist/
#   2. Python runtime for the API (uvicorn + alembic migrations)
#   3. Nginx reverse proxy that serves the built frontend and forwards
#      /api/* to the Python container on port 8000.
#
# Repository code is NEVER executed at runtime; ingestion operates on a cloned
# working copy only.

# ---------- Stage 1: frontend build ----------
FROM node:20-alpine AS frontend-build
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------- Stage 2: backend ----------
FROM python:3.12-slim AS backend

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300

WORKDIR /app

# Install OS build dependencies required by some wheels (e.g. sentence-transformers).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Pre-install PyPI dependencies from pyproject.toml in a cached Docker layer.
# This prevents re-downloading heavy wheels (e.g. torch, sentence-transformers)
# whenever application source code changes.
COPY pyproject.toml ./
RUN mkdir -p repointel packages apps evals \
    && touch repointel/__init__.py packages/__init__.py apps/__init__.py evals/__init__.py \
    && pip install --no-cache-dir -e .

# Copy application source code
COPY repointel/ ./repointel/
COPY packages/ ./packages/
COPY apps/ ./apps/
COPY evals/ ./evals/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY .env.example .env.example

# Register the updated project package without re-downloading dependencies
RUN pip install --no-cache-dir --no-deps -e .

# Entry: migrate then start API
COPY scripts/docker-entrypoint.sh /app/scripts/docker-entrypoint.sh
RUN chmod +x /app/scripts/docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=10 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

CMD ["/app/scripts/docker-entrypoint.sh"]

# ---------- Stage 3: nginx reverse proxy for frontend + API ----------
FROM nginx:1.27-alpine AS frontend

COPY --from=frontend-build /build/web/dist/ /usr/share/nginx/html/
COPY nginx/repointel.conf /etc/nginx/conf.d/default.conf

EXPOSE 80