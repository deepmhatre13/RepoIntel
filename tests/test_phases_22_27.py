import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from packages.knowledge.models import Base, SymbolHistoryModel, SymbolEdgeModel
from packages.knowledge.vector_store import VectorStore
from packages.knowledge.repository_facade import RepositoryFacade
from packages.parsing.test_parser import map_tests_to_production_symbols, DiscoveredTestCase
from packages.parsing.python_ast import SymbolNode

@pytest_asyncio.fixture
async def async_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()

@pytest.mark.asyncio
async def test_symbol_history_facade(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    sh = SymbolHistoryModel(
        repo_id="r1", symbol_name="auth_user", file_path="auth.py", commit_sha="abc12345", author="Alice", change_type="MODIFIED"
    )
    async_session.add(sh)
    await async_session.commit()

    hist = await facade.get_symbol_history("r1", "auth_user")
    assert len(hist) == 1
    assert hist[0]["commit_sha"] == "abc12345"

@pytest.mark.asyncio
async def test_test_discovery_mapping():
    tc = DiscoveredTestCase(file_path="tests/test_auth.py", test_name="test_authenticate_user", imported_modules=["auth"])
    sym = SymbolNode(file_path="auth.py", symbol_name="authenticate_user", symbol_type="function", line_start=1, line_end=10, signature="def authenticate_user()")

    mappings = map_tests_to_production_symbols([tc], [sym])
    assert len(mappings) == 1
    assert mappings[0].test_name == "test_authenticate_user"
    assert mappings[0].target_symbol == "authenticate_user"



@pytest.mark.asyncio
async def test_architecture_snapshot(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    snap = await facade.create_architecture_snapshot("r1", "sha111")
    assert snap["repo_id"] == "r1"
    assert snap["commit_sha"] == "sha111"


@pytest.mark.asyncio
async def test_module_coupling_facade(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    # Single edge crossing two distinct modules (auth -> billing)
    edge = SymbolEdgeModel(
        repo_id="r1",
        source_symbol="auth.user",
        source_file="auth/user.py",
        edge_type="calls",
        target_symbol="billing.pay",
        target_file="billing/pay.py",
    )
    async_session.add(edge)
    await async_session.commit()

    data = await facade.get_module_coupling("r1")
    assert "modules" in data
    assert data["modules"]["auth"]["fan_out"] == 1
    assert data["modules"]["billing"]["fan_in"] == 1
    assert data["modules"]["auth"]["fan_in"] == 0
