# RepoIntel

Evidence-grounded software engineering intelligence platform.

RepoIntel ingests a repository, parses it statically (no code execution), builds a
queryable knowledge layer (relational + graph + retrieval), and answers engineering
questions with claims that are distinguishable as **facts**, **inferences**, and
**recommendations** — each backed by verifiable citations into the analyzed code.

## Problem Statement

Generic LLM assistants hallucinate when asked about a specific codebase: they invent
files, misattribute behavior, and cannot relate a change to its blast radius. RepoIntel
solves this by grounding every generated statement in deterministic evidence extracted
from the repository itself:

```
real repository → safe ingestion → static parsing → versioned database
      ↓
incremental knowledge (fingerprints + scoped invalidation)
      ↓
vector + graph retrieval → Gemini + LangGraph orchestration
      ↓
evidence validation → CLI / API → auditable result
```

## Key Capabilities

| Area | What it does |
|---|---|
| Ingestion | Local paths or git URLs; `.gitignore`-aware filtering; binary exclusion; archetype detection |
| Parsing | Python AST, JavaScript/TypeScript, dependency manifests, symbol extraction, AST-aware chunking |
| Knowledge layer | SQLAlchemy async models; symbols, edges, chunks, commits, tests, architecture snapshots |
| Retrieval | Hybrid vector/keyword search with reranking and context assembly |
| Graph | Symbol call/import graph, traversal, coupling, impact analysis |
| Git intelligence | Commit history, per-symbol change history, co-change relations, API change detection |
| Agentic workflow | LangGraph planner → retriever → synthesis → reviewer → reflection loop, human-in-the-loop |
| LLM | Google Gemini only (`langchain-google-genai`); deterministic degraded mode without an API key |
| Lifecycle | Deterministic state machine (CREATED→…→READY/PARTIAL/FAILED), persisted per run |
| Incremental indexing | Per-file content fingerprints; only changed files re-parsed; scoped cache invalidation |
| Recovery | Error classification, retry with backoff + jitter, cooperative cancellation, stage timeouts |
| Resource budgets | File count/size, total size, parse workers, embedding batch, graph bounds, token budget, LLM call cap, execution deadline |

## Architecture

```
apps/
  cli.py            CLI entry point (python -m repointel.cli)
  api.py            FastAPI application
  job_runner.py     Background analysis jobs driven by the pipeline orchestrator
packages/
  ingestion/        cloning, filtering, metadata, git history
  parsing/          python/js-ts AST, chunking, symbol graph, test discovery
  knowledge/        models, persistence, migrations, facade capabilities,
                    semantic search, unified graph, health/risk/decision engines
  agentic/          Gemini client, LangGraph graphs, agents, copilot backend
  runtime/          pipeline orchestrator, lifecycle, incremental indexer,
                    resource manager, failure recovery
  shared/           telemetry, common schemas
evals/              evaluation suite + benchmark datasets
alembic/            versioned database migrations
tests/              unit, integration, migration and end-to-end tests
```

`RepositoryFacade` is the single public knowledge boundary; agents and services depend
only on it. Capability implementations live in focused mixins under
`packages/knowledge/facade/`. Runtime infrastructure (orchestrator, lifecycle,
recovery, resource management) lives outside the facade by design.

## Technology Stack

Python 3.12 · SQLAlchemy 2.0 (async) · Alembic · FastAPI · Pydantic v2 ·
LangGraph / LangChain Core · `langchain-google-genai` (Gemini) · GitPython ·
Rich (CLI) · pytest / pytest-asyncio

## Installation

```bash
git clone <your-fork-url> repointel
cd repointel
python -m pip install -e ".[dev]"
```

Requires Python ≥ 3.12.
## Environment Variables

Copy `.env.example` to `.env` and fill in values. **Never commit `.env`.**

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | Async SQLAlchemy URL | `sqlite+aiosqlite:///./repointel.db` |
| `GOOGLE_API_KEY` | Gemini API key | *(none — degraded mode)* |
| `GEMINI_MODEL` | Gemini model id | `gemini-1.5-pro` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `MAX_REPOSITORY_FILES` | File-count budget | `20000` |
| `MAX_FILE_SIZE` | Per-file size budget (bytes) | `2097152` |
| `MAX_TOTAL_REPOSITORY_SIZE` | Total size budget (bytes) | `524288000` |
| `MAX_PARSE_WORKERS` | Bounded parse concurrency | `4` |
| `MAX_EMBEDDING_BATCH` | Embedding batch size | `64` |
| `MAX_GRAPH_NODES` / `MAX_GRAPH_EDGES` | Traversal bounds | `100` / `500` |
| `MAX_SEARCH_RESULTS` | Result cap | `20` |
| `MAX_HISTORY_COMMITS` | Git depth cap | `50` |
| `MAX_CONTEXT_TOKENS` | Context budget for prompts | `8000` |
| `MAX_LLM_CALLS` | Hard LLM call cap per run | `25` |
| `MAX_EXECUTION_SECONDS` | Run execution deadline | `900` |

## Database Setup & Migrations

Migrations are the single sanctioned path for schema creation **and** evolution.
`create_all()` is intentionally not used at runtime because it cannot migrate
existing tables — that is the exact class of defect the migration system exists
to prevent.

```bash
python -m repointel.cli db-upgrade      # apply all pending migrations
python -m repointel.cli db-current      # current vs latest revision
python -m repointel.cli db-check        # full schema validation
python -m repointel.cli db-downgrade    # revert the latest migration step
```

`db-check` prints a structured verdict and exits non-zero when the deployed schema
is not current:

```
## CLI Usage

```bash
python -m repointel.cli analyze <path-or-url> [artifact_type]   # full analysis + artifact
python -m repointel.cli status <repo>                           # lifecycle status
python -m repointel.cli query <repo> "<question>"               # evidence-grounded QA
python -m repointel.cli update <repo>                           # incremental re-analysis
python -m repointel.cli reindex <repo> --force                  # forced full rebuild
python -m repointel.cli history <repo> <symbol> [limit]
python -m repointel.cli diff <repo> <commit_sha>
python -m repointel.cli architecture <repo>
python -m repointel.cli hotspots <repo> [limit]
python -m repointel.cli coupling <repo> [file]
python -m repointel.cli tests <repo> <symbol>
python -m repointel.cli impact <repo> <target>
python -m repointel.cli change-plan <repo> <requirement> [target]
python -m repointel.cli eval                                    # retrieval metrics
python -m repointel.cli --help
```

Operational failures print a concise structured error and exit non-zero; pass
`--verbose` to surface full tracebacks during development.

A successful `analyze` reports repository id, run id, status, artifact id/type,
files processed, symbols, chunks, graph relationships, duration, warnings and errors.

## API Usage

```bash
python -m uvicorn apps.api:app --reload
# Interactive docs: http://127.0.0.1:8000/docs  (OpenAPI at /openapi.json)
```

Core routes include repository creation, background `/analyze`, `/runs/{id}/status`,
artifact retrieval, chat/copilot, symbols, dependencies, graph, call-chain, commits,
impact, and repository groups. Startup initializes the schema through migrations via
a modern FastAPI lifespan handler; shutdown disposes engine resources.

## RAG Architecture

Retrieval combines lexical overlap over persisted chunks with graph-derived related
symbols, then reranks evidence before assembling a bounded context window
## Gemini Integration

Gemini is the only LLM provider. All calls flow through
`packages/agentic/llm_client.py`, which records token usage, latency and estimated
cost, and raises typed failures for transient errors. When no API key is configured
or Gemini fails, deterministic capabilities (exact search, symbol lookup, graph
traversal, dependency analysis, git history, impact, risk metrics) keep working and
results are explicitly marked `llm_available=false` — never presented as LLM output.

## Evidence Grounding

Answers separate FACTS / INFERENCES / RECOMMENDATIONS. Every fact must carry evidence
(repository, file, symbol, line range where available). The reviewer performs
deterministic citation verification against indexed files; unsupported or stale
citations force revision rather than approval. Unsupported claims never silently pass.

## Incremental Indexing

Each analyzed file leaves a content fingerprint. Subsequent `update` runs diff
current vs stored fingerprints into ADDED/MODIFIED/DELETED/RENAMED/UNCHANGED sets,
expand affected files across the symbol graph, invalidate only impacted rows, and
re-embed only what changed. Verified behavior: an unchanged repository reports
`0 files parsed` and completes in roughly half the cold time.

## Evaluation

```bash
python -m repointel.cli eval          # writes ./evals/reports/evaluation_report.json
python -m pytest                      # unit + integration + migration + E2E suites
```

Metrics include Recall@K, Precision@K and MRR over expected files, plus claim-support
ratios for grounding quality. Baselines (naive LLM, simple vector RAG) live in
## Limitations

- Without `GOOGLE_API_KEY`, LLM synthesis runs in deterministic degraded mode;
  answers are explicitly marked as such and carry lower confidence.
- The bundled embedding service returns deterministic placeholder vectors; swap in a
  real model behind `EmbeddingService.embed_texts` for semantic-quality retrieval.
- The Starlette testclient emits a third-party deprecation notice about `httpx`
  versus `httpx2`; it is outside this project's control until upstream resolves it.
- Migration downgrade for schema reconciliation is forward-only by design (data
  protection); only additive column drops are reversible.
- Windows path handling is normalized internally, but long-path limits can affect
  deeply nested temporary clones.

## Security Model

- **Static analysis only.** Repository code is never executed: no `setup.py`,
  package installation, npm lifecycle scripts, Makefiles or CI scripts are run.
  Manifests are parsed as data.
- **Secret hygiene.** `.env` is never committed; API keys are read from the
  environment and are excluded from logs, artifacts and evidence payloads.
- **Untrusted input.** Repositories are cloned/read-only inputs; parsing is
  AST-based with bounded resources and no dynamic imports.
- **Auditable output.** Every answer carries citations and run identifiers so
  results can be traced back to source evidence.

## Production Readiness Status

Verified end-to-end (see `tests/test_migrations.py`, `tests/test_e2e_pipeline.py`):

- fresh database migrates to head; existing legacy database upgrades in place with
  data preserved (`repositories.status` defect eliminated)
- repeated FULL analysis is byte-count identical (idempotent)
- incremental update processes only changed files
- forced reindex rebuilds derived knowledge while preserving repository identity
- lifecycle state persists (`READY`) on both repository and run records
- CLI commands validate arguments and return meaningful exit codes

Not yet verified and therefore not claimed: multi-repository benchmarks against
large public repositories under a configured Gemini key, and Postgres deployment.

`evals/baselines.py`.

## Benchmark Methodology

Benchmarks run the full production pipeline against real public repositories of
varying archetype and size, recording measured values only (file/symbol/chunk counts,
indexing time, query latency, LLM calls, tokens, estimated cost). No repository-
specific hacks, hardcoded answers, or fabricated numbers are used; failures are
reported as failures.

(`MAX_CONTEXT_TOKENS`). Retrieved items carry file path, line range and relevance so
the synthesizer can cite them precisely.

## Knowledge Graph

The unified graph links repositories, files, symbols and their call/import edges.
Traversal is bounded by `MAX_GRAPH_NODES`/`MAX_GRAPH_EDGES`; impact analysis expands
dependents transitively with hop limits.

## LangGraph Workflow

`Planner → Retriever → Reflection(Synthesis ↔ Reviewer) → Finalize`, with conditional
routing on the reviewer verdict (`APPROVE`, `REVISE_SYNTHESIS`, `REVISE_PLAN`,
`NEEDS_HUMAN_REVIEW`) and a bounded reflection iteration budget.

Database: OK
Migration: CURRENT
Tables: OK
Columns: OK
Indexes: OK
Foreign Keys: OK
Database validation PASSED.
```

Existing pre-migration databases upgrade in place: historical rows are preserved
and new columns receive safe defaults (`status='CREATED'`, `branch=''`,
`updated_at` initialized from `created_at`).

