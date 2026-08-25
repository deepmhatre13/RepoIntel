import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from packages.knowledge.models import Base
from packages.knowledge.vector_store import VectorStore
from packages.knowledge.repository_facade import RepositoryFacade
from packages.agentic.state.schema import RepoIntelState

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
async def test_repository_group(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)
    
    group = await facade.create_repository_group("grp-1", "Services Group", "Backend microservices")
    assert group["id"] == "grp-1"
    
    await facade.add_repo_to_group("grp-1", "repo-a")
    await facade.add_repo_to_group("grp-1", "repo-b")
    
    await facade.add_cross_repo_relation("repo-a", "repo-b", "HTTP_API", "REST call from service A to B")
    neighbors = await facade.cross_repository_neighbors("repo-a")
    assert len(neighbors) == 1
    assert neighbors[0]["target_repo_id"] == "repo-b"

@pytest.mark.asyncio
async def test_call_chain_and_impact(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)
    
    chain = await facade.trace_call_chain("repo-1", "auth_user")
    assert isinstance(chain, list)

    impact = await facade.analyze_impact("repo-1", "auth_user")
    assert impact["target"] == "auth_user"
    assert impact["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

@pytest.mark.asyncio
async def test_state_schema():
    state = RepoIntelState(
        run_id="run-1",
        thread_id="t-1",
        repo_id="r-1",
        repo_url="https://github.com/example/repo",
        archetype="SERVICE",
        artifact_type="ARCHITECTURE",
        active_repository_ids=["r-1", "r-2"]
    )
    assert len(state.active_repository_ids) == 2
    assert state.artifact_type == "ARCHITECTURE"
