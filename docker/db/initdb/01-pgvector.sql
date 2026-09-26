-- RepoIntel PostgreSQL init: ensure the pgvector `vector` extension exists.
-- Runs automatically by the PostgreSQL container entrypoint on a FRESH data
-- directory only (idempotent). The pgvector/pgvector:pg16 base image installs
-- the extension binaries; this statement guarantees it is enabled in the app
-- database before any Alembic migration executes.
CREATE EXTENSION IF NOT EXISTS vector;