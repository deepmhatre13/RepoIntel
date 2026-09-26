import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from packages.knowledge.models import (
    Base,
    SymbolModel,
    DependencyModel,
    RepositoryModel,
)
from packages.knowledge.vector_store import VectorStore
from packages.knowledge.repository_facade import RepositoryFacade
from packages.agentic.query_orchestrator import UnifiedQueryOrchestrator, QueryIntentType, QueryComplexity
from packages.agentic.autonomous_investigator import AutonomousRepositoryInvestigator, SufficiencyStatus
from packages.agentic.copilot import CopilotMode


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
async def test_phase34_query_orchestrator(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)
    orchestrator = UnifiedQueryOrchestrator(facade)

    plan = orchestrator.plan_query("Why is payment service tightly coupled?")
    assert plan.intent == QueryIntentType.INVESTIGATION
    assert plan.complexity == QueryComplexity.INVESTIGATION
    assert "impact" in plan.required_capabilities

    res = await orchestrator.execute_plan("repo1", plan)
    assert res["query"] == plan.query
    assert "orchestrated_answer" in res


@pytest.mark.asyncio
async def test_phase35_autonomous_investigator(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    sym = SymbolModel(repo_id="repo1", symbol_name="process_payment", file_path="payment/service.py", symbol_type="function", line_start=1, line_end=20, signature="def process_payment(): pass")
    async_session.add(sym)
    await async_session.commit()

    investigator = AutonomousRepositoryInvestigator(facade)
    state = await investigator.investigate_autonomously("repo1", "Why is process_payment risky?", max_steps=2)

    assert state.investigation_question == "Why is process_payment risky?"
    assert state.sufficiency_status in (SufficiencyStatus.SUFFICIENT, SufficiencyStatus.INSUFFICIENT, SufficiencyStatus.CONFLICTING)
    assert len(state.facts) >= 1


@pytest.mark.asyncio
async def test_phase36_repository_health_engine(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    repo = RepositoryModel(id="repo1", repo_url="https://github.com/org/repo1", archetype="Python", size_mb=10, file_count=5, manifest_data='{}')
    dep = DependencyModel(repo_id="repo1", package_name="fastapi", ecosystem="PyPI", version_spec="^0.100.0")
    async_session.add_all([repo, dep])
    await async_session.commit()

    health_report = await facade.get_repository_health("repo1")
    assert health_report["repo_id"] == "repo1"
    assert "overall_status" in health_report
    assert len(health_report["signals"]) >= 1


@pytest.mark.asyncio
async def test_phase37_root_cause_analysis(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    sym = SymbolModel(repo_id="repo1", symbol_name="auth_handler", file_path="auth/handler.py", symbol_type="function", line_start=1, line_end=15, signature="def auth_handler(): pass")
    async_session.add(sym)
    await async_session.commit()

    rca = await facade.analyze_root_cause("repo1", "Token validation fails randomly", "BEHAVIORAL")
    assert rca["problem"] == "Token validation fails randomly"
    assert len(rca["candidate_causes"]) >= 1
    assert "evidence_for" in rca["candidate_causes"][0]
    assert "evidence_against" in rca["candidate_causes"][0]


@pytest.mark.asyncio
async def test_phase38_change_simulation(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    sym = SymbolModel(repo_id="repo1", symbol_name="UserStore", file_path="models/user.py", symbol_type="class", line_start=1, line_end=50, signature="class UserStore: pass")
    async_session.add(sym)
    await async_session.commit()

    sim = await facade.simulate_engineering_change("repo1", "Migrate UserStore from SQLite to Postgres", target_entity="UserStore")
    assert sim["proposed_change"] == "Migrate UserStore from SQLite to Postgres"
    assert len(sim["consequences"]) >= 1
    assert any(c["label"] == "CURRENT FACT" for c in sim["consequences"])


@pytest.mark.asyncio
async def test_phase39_copilot_backend(async_session):
    vs = VectorStore()
    facade = RepositoryFacade(async_session, vs)

    req = {
        "repository_id": "repo1",
        "query": "Assess codebase health and architectural coupling",
        "mode": "HEALTH",
    }
    copilot_res = await facade.execute_copilot_request(req)
    assert copilot_res["mode"] == CopilotMode.HEALTH.value
    assert "overall_status" in copilot_res["answer"] or "facts" in copilot_res
