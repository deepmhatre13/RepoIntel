from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from langgraph.types import interrupt
from packages.agentic.state.schema import RepoIntelState, ReviewVerdict, HumanFeedback
from packages.agentic.llm_client import LLMFailureError, llm_node_context
from packages.shared.llm_mode import LLMMode
from packages.agentic.agents.planner.node import planner_node
from packages.agentic.agents.retriever.node import retriever_node
from packages.agentic.graphs.reflection_subgraph import build_reflection_subgraph
from packages.knowledge.persistent_memory import PersistentMemoryStore

def route_main(state: RepoIntelState) -> str:
    if not state.review_report:
        return END

    verdict = state.review_report.verdict
    if verdict == ReviewVerdict.APPROVE:
        return "Finalize"
    elif verdict == ReviewVerdict.REVISE_PLAN:
        return "Planner"
    elif verdict == ReviewVerdict.NEEDS_HUMAN_REVIEW:
        return "HumanReview"
    return "Finalize"

def build_main_graph(llm_client, facade, checkpointer=None, store=None):
    """Build the main LangGraph with optional checkpointing.
    
    Args:
        llm_client: LLM client for agent operations
        facade: Repository facade for knowledge access
        checkpointer: Optional LangGraph checkpoint saver for state persistence.
                     When None, the saver is resolved through
                     ``packages.knowledge.checkpoint_store.resolve_checkpointer``
                     so durable checkpointing is never silently downgraded.
    """
    graph = StateGraph(RepoIntelState)

    async def _planner_step(state: RepoIntelState, runtime: Runtime):
        repository_memory = {}
        if runtime.store is not None:
            memory = PersistentMemoryStore(runtime.store)
            await memory.save_archetype(state.repo_id, state.archetype)
            repository_memory = await memory.load_repository_context(state.repo_id)

        with llm_node_context("Planner"):
            planner_result = planner_node(state, llm_client, repository_memory)
        if hasattr(llm_client, "get_mode"):
            active_mode = llm_client.get_mode()
        elif hasattr(llm_client, "last_mode"):
            active_mode = llm_client.last_mode
        else:
            active_mode = "LIVE_GEMINI" if getattr(llm_client, "is_available", lambda: False)() else "DEGRADED_FALLBACK"
        llm_mode: LLMMode = active_mode
        return {
            **planner_result,
            "llm_mode": llm_mode,
            "degraded": llm_mode != "LIVE_GEMINI",
        }

    async def _retriever_step(state: RepoIntelState):
        with llm_node_context("Retriever"):
            return await retriever_node(state, facade, llm_client)

    reflection_subgraph = build_reflection_subgraph(llm_client, facade)

    def _finalize_step(state: RepoIntelState):
        return {"route_decision": "FINALIZED"}

    async def _human_review_step(state: RepoIntelState, runtime: Runtime):
        """Human review node using LangGraph's native interrupt()/Command(resume=...) HITL.

        First invocation: interrupt() persists a checkpoint, suspends the graph,
        and surfaces the review payload to the caller. Execution never proceeds
        past this point until the graph is resumed with Command(resume=<payload>).

        Resumed invocation: LangGraph re-executes this node from the checkpoint,
        and interrupt() returns the resume payload (the human's decision). The
        decision is written into state.human_feedback as a durable state update
        and the routing decision is derived from it — the human's input is the
        single source of truth, not a database boolean.
        """
        # If human feedback is already present in state (e.g. replayed
        # checkpoint after the decision was applied), route on it directly.
        if state.human_feedback:
            if runtime.store is not None:
                await PersistentMemoryStore(runtime.store).save_human_correction(
                    state.repo_id, state.run_id, state.human_feedback
                )
            return _route_on_human_feedback(state.human_feedback)

        # First pass: suspend and request human input.
        resume_payload = interrupt({
            "type": "human_review_required",
            "run_id": state.run_id,
            "thread_id": state.thread_id,
            "review_report": state.review_report.model_dump() if state.review_report else None,
            "message": "Human review required before continuing analysis.",
        })

        # Resumed: interrupt() returned the human's decision payload.
        if isinstance(resume_payload, HumanFeedback):
            feedback = resume_payload
        elif isinstance(resume_payload, dict):
            feedback = HumanFeedback(
                approved=bool(resume_payload.get("approved", False)),
                decision=resume_payload.get("decision", "REJECT"),
                comments=resume_payload.get("comments"),
            )
        else:
            # Resume without a usable decision — remain suspended for review.
            return {"route_decision": "NEEDS_HUMAN_REVIEW"}
        if runtime.store is not None:
            await PersistentMemoryStore(runtime.store).save_human_correction(
                state.repo_id, state.run_id, feedback
            )
        return _route_on_human_feedback(feedback)

    def _route_on_human_feedback(feedback: HumanFeedback) -> Dict[str, Any]:
        # The decision is applied and then cleared from state: keeping it set
        # would make every later HumanReview pass take the "already decided"
        # branch and route back into reflection, so a reviewer that keeps
        # returning NEEDS_HUMAN_REVIEW would loop forever without ever
        # suspending again. Clearing it makes the next HumanReview pass call
        # interrupt() and suspend for a fresh human decision.
        if feedback.decision == "APPROVE":
            return {"human_feedback": None, "route_decision": "FINALIZED"}
        if feedback.decision == "REQUEST_REVISION":
            return {"human_feedback": None, "route_decision": "REVISE_SYNTHESIS"}
        # Unknown/REJECT decision — require another human pass.
        return {"human_feedback": None, "route_decision": "NEEDS_HUMAN_REVIEW"}

    graph.add_node("Planner", _planner_step)
    graph.add_node(
        "Retriever",
        _retriever_step,
        retry_policy=RetryPolicy(max_attempts=2, retry_on=(LLMFailureError,)),
    )
    graph.add_node("ReflectionSubgraph", reflection_subgraph)
    graph.add_node("Finalize", _finalize_step)
    graph.add_node("HumanReview", _human_review_step)

    graph.set_entry_point("Planner")
    graph.add_edge("Planner", "Retriever")
    graph.add_edge("Retriever", "ReflectionSubgraph")

    graph.add_conditional_edges(
        "ReflectionSubgraph",
        route_main,
        {
            "Finalize": "Finalize",
            "Planner": "Planner",
            "HumanReview": "HumanReview"
        }
    )

    graph.add_conditional_edges(
        "HumanReview",
        lambda s: "Finalize" if s.route_decision == "FINALIZED" else ("ReflectionSubgraph" if s.route_decision == "REVISE_SYNTHESIS" else END),
        {
            "Finalize": "Finalize",
            "ReflectionSubgraph": "ReflectionSubgraph",
            END: END
        }
    )

    graph.add_edge("Finalize", END)

    # Durable checkpointing is a correctness requirement, not a best-effort
    # enhancement: a run that silently checkpointed into process memory cannot be
    # recovered after a worker restart, and an operator would have no signal that
    # recovery was lost. The saver is therefore constructed through an explicit
    # provider (``REPOINTEL_ALLOW_INMEMORY_CHECKPOINTS=1`` permits the in-memory
    # saver for tests/CLI only) and never as a blanket ``except Exception``
    # fallback that hides an unusable database.
    if checkpointer is None:
        from packages.knowledge.checkpoint_store import resolve_checkpointer

        checkpointer = resolve_checkpointer()

    return graph.compile(checkpointer=checkpointer, store=store)
