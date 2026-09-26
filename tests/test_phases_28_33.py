import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from packages.knowledge.models import (
    Base,
    SymbolModel,
    SymbolEdgeModel,
    RepositoryModel,
)
from packages.knowledge.vector_store import VectorStore
from packages.knowledge.repository_facade import RepositoryFacade
from packages.agentic.investigation_agent import EngineeringInvestigationAgent
from packages.agentic.report_generator import ReportType


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
async def test_phase28_unified_knowledge_graph(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    repo = RepositoryModel(id="repo1", repo_url="https://github.com/org/repo1", archetype="Python", size_mb=10, file_count=5, manifest_data='{"languages": {"Python": 100}}')
    sym1 = SymbolModel(repo_id="repo1", symbol_name="authenticate", file_path="auth/service.py", symbol_type="function", line_start=1, line_end=20, signature="def authenticate(): pass")
    sym2 = SymbolModel(repo_id="repo1", symbol_name="get_user", file_path="db/user_repo.py", symbol_type="function", line_start=1, line_end=15, signature="def get_user(): pass")
    edge = SymbolEdgeModel(repo_id="repo1", source_symbol="authenticate", source_file="auth/service.py", edge_type="calls", target_symbol="get_user", target_file="db/user_repo.py")

    async_session.add_all([repo, sym1, sym2, edge])
    await async_session.commit()

    kg = await facade.get_knowledge_graph("repo1")
    assert len(kg._nodes) >= 3  # repo, file, symbols

    res = await facade.query_knowledge_graph("repo1", start_node_id="symbol:auth/service.py:authenticate", action="traverse")
    assert res["node_count"] >= 1


@pytest.mark.asyncio
async def test_phase29_semantic_code_search(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    sym = SymbolModel(repo_id="repo1", symbol_name="validate_token", file_path="auth/token.py", symbol_type="function", line_start=10, line_end=30, signature="def validate_token(): pass")
    async_session.add(sym)
    await async_session.commit()

    res = await facade.search_code("repo1", query="validate_token", intent="symbol", mode="exact")
    assert res["query"] == "validate_token"
    assert len(res["results"]) >= 1
    assert res["results"][0]["symbol"] == "validate_token"


@pytest.mark.asyncio
async def test_phase30_software_architecture_analyst(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    sym1 = SymbolModel(repo_id="repo1", symbol_name="login_endpoint", file_path="api/routes.py", symbol_type="function", line_start=1, line_end=10, signature="def login_endpoint(): pass")
    sym2 = SymbolModel(repo_id="repo1", symbol_name="auth_service", file_path="services/auth.py", symbol_type="function", line_start=1, line_end=10, signature="def auth_service(): pass")
    edge = SymbolEdgeModel(repo_id="repo1", source_symbol="login_endpoint", source_file="api/routes.py", edge_type="calls", target_symbol="auth_service", target_file="services/auth.py")

    async_session.add_all([sym1, sym2, edge])
    await async_session.commit()

    arch = await facade.analyze_architecture("repo1")
    assert "facts" in arch
    assert "inferences" in arch
    assert "recommendations" in arch
    assert "components" in arch


@pytest.mark.asyncio
async def test_phase31_investigation_agent(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    sym = SymbolModel(repo_id="repo1", symbol_name="db_query", file_path="db/client.py", symbol_type="function", line_start=1, line_end=5, signature="def db_query(): pass")
    async_session.add(sym)
    await async_session.commit()

    agent = EngineeringInvestigationAgent(facade=facade)
    report = await agent.investigate("repo1", "Why is db_query heavily coupled?")

    assert report.question == "Why is db_query heavily coupled?"
    assert len(report.facts) >= 1
    assert len(report.recommendations) >= 1


@pytest.mark.asyncio
async def test_phase32_decision_intelligence(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    decision = await facade.evaluate_engineering_decision("repo1", "Should we extract authentication service?", target_entity="auth")
    assert "decision" in decision
    assert len(decision["alternatives"]) == 3
    assert len(decision["facts"]) >= 1


@pytest.mark.asyncio
async def test_phase33_report_generator(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    repo = RepositoryModel(id="repo1", repo_url="https://github.com/org/repo1", archetype="Python", size_mb=5, file_count=3, manifest_data='{}')
    async_session.add(repo)
    await async_session.commit()

    rep1 = await facade.generate_engineering_report("repo1", "ENGINEERING_OVERVIEW")
    assert rep1["report_type"] == ReportType.ENGINEERING_OVERVIEW.value
    assert len(rep1["sections"]) >= 1

    rep2 = await facade.generate_engineering_report("repo1", "ARCHITECTURE_REPORT")
    assert rep2["report_type"] == ReportType.ARCHITECTURE_REPORT.value
