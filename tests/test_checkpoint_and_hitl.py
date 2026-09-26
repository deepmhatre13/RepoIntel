"""Targeted test suite for LangGraph checkpointing, process recovery, and native HITL interrupt/resume.
"""
import pytest
import uuid
from packages.agentic.state.schema import (
    RepoIntelState,
    ReviewReport,
    ReviewVerdict,
    HumanFeedback,
    Plan,
    ArtifactDraft,
)
from packages.agentic.graphs.phase1_main_graph import build_main_graph
from langgraph.checkpoint.memory import MemorySaver


class DummyLLMClient:
    def complete_structured(self, prompt, schema, system=None):
        if schema == Plan:
            return Plan(plan_id="plan-1", repo_id="repo-1", artifact_type="README", tasks=[])
        if schema == ReviewReport:
            return ReviewReport(verdict=ReviewVerdict.NEEDS_HUMAN_REVIEW, summary="Needs review")
        if schema == ArtifactDraft:
            return ArtifactDraft(artifact_type="README", sections=[])
        return schema()

class DummyFacade:
    pass


@pytest.mark.asyncio
async def test_hitl_interrupt_and_resume_approve():
    """Verify native LangGraph interrupt on HumanReview and resume with APPROVE decision."""
    llm_client = DummyLLMClient()
    facade = DummyFacade()
    checkpointer = MemorySaver()
    graph = build_main_graph(llm_client, facade, checkpointer=checkpointer)

    thread_id = f"thread-hitl-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    # Initial state with NEEDS_HUMAN_REVIEW
    initial_state = RepoIntelState(
        run_id="run-1",
        thread_id=thread_id,
        repo_id="repo-1",
        repo_url="https://github.com/example/repo",
        archetype="SERVICE",
        review_report=ReviewReport(verdict=ReviewVerdict.NEEDS_HUMAN_REVIEW, summary="Requires human review"),
    )

    # 1. First invocation — should trigger interrupt at HumanReview
    res1 = await graph.ainvoke(initial_state, config=config)
    assert res1.get("route_decision") == "NEEDS_HUMAN_REVIEW"

    # 2. Check checkpoint state stored
    state_snapshot = await graph.aget_state(config)
    assert state_snapshot is not None

    # 3. Resume with APPROVE decision
    state_dict = dict(state_snapshot.values)
    state_dict["human_feedback"] = HumanFeedback(approved=True, decision="APPROVE", comments="Approved by reviewer")
    resumed_state = RepoIntelState(**state_dict)

    res2 = await graph.ainvoke(resumed_state, config=config)
    assert res2.get("route_decision") == "FINALIZED"


@pytest.mark.asyncio
async def test_hitl_interrupt_and_resume_request_revision():
    """Verify native LangGraph interrupt on HumanReview and resume with REQUEST_REVISION decision."""
    llm_client = DummyLLMClient()
    facade = DummyFacade()
    checkpointer = MemorySaver()
    graph = build_main_graph(llm_client, facade, checkpointer=checkpointer)

    thread_id = f"thread-hitl-rev-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = RepoIntelState(
        run_id="run-2",
        thread_id=thread_id,
        repo_id="repo-1",
        repo_url="https://github.com/example/repo",
        archetype="SERVICE",
        review_report=ReviewReport(verdict=ReviewVerdict.NEEDS_HUMAN_REVIEW, summary="Requires revision"),
    )

    res1 = await graph.ainvoke(initial_state, config=config)
    assert res1.get("route_decision") == "NEEDS_HUMAN_REVIEW"

    # Resume with REQUEST_REVISION decision
    state_dict = initial_state.model_dump()
    state_dict["human_feedback"] = HumanFeedback(approved=False, decision="REQUEST_REVISION", comments="Please revise synthesis")
    resumed_state = RepoIntelState(**state_dict)

    res2 = await graph.ainvoke(resumed_state, config=config)
    # The graph routes HumanReview -> ReflectionSubgraph (Synthesis -> Reviewer).
    # Since DummyLLMClient Reviewer returns NEEDS_HUMAN_REVIEW, the graph completes
    # reflection and pauses at HumanReview again for the next review cycle.
    assert res2.get("route_decision") == "NEEDS_HUMAN_REVIEW"
    assert res2.get("reflection_iteration_count", 0) >= 1


@pytest.mark.asyncio
async def test_process_recovery_simulation():
    """Verify process recovery: state checkpointed in graph instance 1 can be restored in graph instance 2."""
    llm_client = DummyLLMClient()
    facade = DummyFacade()
    checkpointer = MemorySaver()

    thread_id = f"thread-recovery-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    # Process 1: run graph instance 1
    graph1 = build_main_graph(llm_client, facade, checkpointer=checkpointer)
    initial_state = RepoIntelState(
        run_id="run-recovery-1",
        thread_id=thread_id,
        repo_id="repo-rec-1",
        repo_url="https://github.com/example/repo",
        archetype="SERVICE",
        review_report=ReviewReport(verdict=ReviewVerdict.NEEDS_HUMAN_REVIEW),
    )
    await graph1.ainvoke(initial_state, config=config)

    # Process 2: instantiate brand-new graph object using same checkpointer
    graph2 = build_main_graph(llm_client, facade, checkpointer=checkpointer)
    saved_state = await graph2.aget_state(config)
    assert saved_state is not None
    assert saved_state.values["run_id"] == "run-recovery-1"

    # Process 2 resumes execution
    state_dict = dict(saved_state.values)
    state_dict["human_feedback"] = HumanFeedback(approved=True, decision="APPROVE")
    resumed_state = RepoIntelState(**state_dict)
    final_res = await graph2.ainvoke(resumed_state, config=config)
    assert final_res.get("route_decision") == "FINALIZED"


@pytest.mark.asyncio
async def test_database_checkpoint_native_resume(tmp_path, monkeypatch):
    """A new saver/graph must resume a durable interrupt without rerunning START."""
    from typing import TypedDict
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Command, interrupt
    from packages.agentic.checkpoint_store import PostgresCheckpointer

    class State(TypedDict):
        count: int
        approved: bool

    def prepare(state):
        return {"count": state["count"] + 1}

    def review(state):
        return {"approved": interrupt({"count": state["count"]})["approved"]}

    def build(saver):
        builder = StateGraph(State)
        builder.add_node("prepare", prepare)
        builder.add_node("review", review)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "review")
        builder.add_edge("review", END)
        return builder.compile(checkpointer=saver)

    url = f"sqlite+aiosqlite:///{(tmp_path / 'checkpoints.db').as_posix()}"
    config = {"configurable": {"thread_id": "durable-review"}}
    from packages.knowledge import migrations
    monkeypatch.setenv("DATABASE_URL", url)
    migrations.upgrade_to_head()
    engine = create_async_engine(url)
    try:
        graph = build(PostgresCheckpointer(async_sessionmaker(engine)))
        await graph.ainvoke({"count": 0, "approved": False}, config)
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("review",)
        assert snapshot.tasks[0].interrupts
    finally:
        await engine.dispose()

    # No saver, connection pool or graph object is reused.
    engine = create_async_engine(url)
    try:
        saver = PostgresCheckpointer(async_sessionmaker(engine))
        graph = build(saver)
        result = await graph.ainvoke(Command(resume={"approved": True}), config)
        assert result == {"count": 1, "approved": True}
        assert (await graph.aget_state(config)).next == ()
        history = [snapshot async for snapshot in graph.aget_state_history(config)]
        assert len(history) >= 4
    finally:
        await engine.dispose()


