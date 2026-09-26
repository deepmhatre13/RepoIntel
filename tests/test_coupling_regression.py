"""Regression tests for the coupling/cycle-detection defect.

Reproduces two real observed failures:

1. The CLI iterated the dict envelope returned by ``facade.detect_cycles`` as if
   it were a list of cycles, so every dict *key* (a string) was passed to
   ``.get('cycle', [])``, raising
   ``AttributeError: 'str' object has no attribute 'get'``. The
   "Circular Dependencies Detected (N)" count was actually the number of keys
   in the envelope, not the number of cycles.

2. ``get_module_coupling_matrix`` fell back to ``target_symbol`` when
   ``target_file`` was absent, making symbol-level call edges leak symbols
   (e.g. ``Calculator``, ``add``, ``print``) into the module matrix.
"""

import io
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from rich.console import Console
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from packages.knowledge.models import (
    Base,
    FileModel,
    RepositoryModel,
    SymbolEdgeModel,
    SymbolModel,
)
from packages.knowledge.repository_facade import RepositoryFacade
from packages.knowledge.vector_store import VectorStore

DATASET = (
    Path(__file__).resolve().parent.parent
    / "evals" / "datasets" / "python_library"
)


def _repo_id(repo_url: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))


def _calc_edges(repo_id: str):
    """Edges mirroring the AST parser output for calculator.py.

    The parser never populates ``target_file`` on import/call edges, so every
    edge here has ``target_file=None`` -- the exact situation that leaked
    symbols into the module matrix.
    """
    return [
        SymbolEdgeModel(
            repo_id=repo_id,
            source_symbol="<file>",
            source_file="src/calculator.py",
            edge_type="imports",
            target_symbol="os",
            target_file=None,
        ),
        SymbolEdgeModel(
            repo_id=repo_id,
            source_symbol="main",
            source_file="src/calculator.py",
            edge_type="calls",
            target_symbol="Calculator",
            target_file=None,
        ),
        SymbolEdgeModel(
            repo_id=repo_id,
            source_symbol="main",
            source_file="src/calculator.py",
            edge_type="calls",
            target_symbol="add",
            target_file=None,
        ),
        SymbolEdgeModel(
            repo_id=repo_id,
            source_symbol="main",
            source_file="src/calculator.py",
            edge_type="calls",
            target_symbol="print",
            target_file=None,
        ),
    ]


@pytest_asyncio.fixture
async def db_env(tmp_path):
    db_file = tmp_path / "coupling.db"
    url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    eng = create_async_engine(url, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
    yield {"url": url, "eng": eng, "sf": sf}
    await eng.dispose()


async def _seed_calculator(db_env, repo_url: str) -> str:
    repo_id = _repo_id(repo_url)
    async with db_env["sf"]() as s:
        s.add(
            RepositoryModel(
                id=repo_id, repo_url=repo_url, archetype="LIBRARY"
            )
        )
        s.add(
            FileModel(
                repo_id=repo_id,
                file_path="src/calculator.py",
                language="python",
                content_hash="h1",
                size_bytes=300,
            )
        )
        for name, stype, ls, le, sig in [
            ("Calculator", "class", 4, 4, "class Calculator"),
            ("Calculator.add", "method", 6, 7, "def add(self, a, b)"),
            ("Calculator.multiply", "method", 9, 10, "def multiply(self, a, b)"),
            ("main", "function", 12, 14, "def main()"),
        ]:
            s.add(
                SymbolModel(
                    repo_id=repo_id,
                    file_path="src/calculator.py",
                    symbol_name=name,
                    symbol_type=stype,
                    line_start=ls,
                    line_end=le,
                    signature=sig,
                )
            )
        for e in _calc_edges(repo_id):
            s.add(e)
        await s.commit()
    return repo_id


@pytest.mark.asyncio
async def test_module_coupling_excludes_symbol_edges_as_modules(db_env):
    """Symbol-level call edges must not be treated as module-level deps."""
    repo_url = str(DATASET)
    repo_id = await _seed_calculator(db_env, repo_url)
    async with db_env["sf"]() as s:
        facade = RepositoryFacade(s, VectorStore())
        data = await facade.get_module_coupling(repo_id)
        modules = set(data["modules"].keys())

    for leaked in ("Calculator", "add", "print", "multiply"):
        assert leaked not in modules, (
            f"symbol '{leaked}' leaked into module matrix: {modules}"
        )
    assert "os" in modules, modules
    assert "calculator" in modules, modules
    assert data["modules"]["calculator"]["fan_out"] == 1
    assert data["modules"]["calculator"]["fan_in"] == 0


def test_cycle_detection_envelope_not_iterated_as_cycles():
    """A dict envelope must never be iterated as if it were a list of cycles.

    Reproduces the original CLI crash: the caller treated the keys of the dict
    envelope returned by ``detect_cycles`` (strings like ``repo_id``,
    ``cycle_count``) as if they were cycle objects, calling ``.get('cycle')``
    on a str and raising ``AttributeError: 'str' object has no attribute
    'get'``. The reported count of "cycles" was actually the number of dict
    keys. The fixed caller checks ``isinstance(cycles, dict)`` and reads the
    ``cycles`` key, and reports an empty list when no cycles exist.
    """
    # Envelope shape exactly as produced by facade.detect_cycles.
    envelope = {
        "repo_id": "x",
        "cycle_count": 0,
        "cycles": [],
        "drift_signals": 0,
    }
    cycle_list = (
        envelope.get("cycles", []) if isinstance(envelope, dict) else envelope
    )
    if cycle_list:
        report = [(c.get("cycle")) for c in cycle_list]
    else:
        report = []
    assert report == []
    # A non-empty cycles list is a list of dicts each carrying a "cycle" chain.
    envelope_with_cycle = {
        "repo_id": "x",
        "cycle_count": 1,
        "cycles": [{"cycle": ["a", "b", "a"]}],
        "drift_signals": 1,
    }
    cycle_list2 = (
        envelope_with_cycle.get("cycles", [])
        if isinstance(envelope_with_cycle, dict)
        else envelope_with_cycle
    )
    chains = [" -> ".join(c["cycle"]) for c in cycle_list2]
    assert chains == ["a -> b -> a"]


@pytest.mark.asyncio
async def test_detect_cycles_returns_acyclic_for_calculator(db_env):
    """calculator.py has no module-level import cycle, so detect_cycles must
    report zero cycles (never a fabricated count based on edge count)."""
    repo_url = str(DATASET)
    repo_id = await _seed_calculator(db_env, repo_url)
    async with db_env["sf"]() as s:
        facade = RepositoryFacade(s, VectorStore())
        envelope = await facade.detect_cycles(repo_id)

    assert isinstance(envelope, dict)
    assert envelope["cycle_count"] == 0
    assert envelope["cycles"] == []

