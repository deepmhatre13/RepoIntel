from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from packages.agentic.state.schema import RepoIntelState, ReviewVerdict
from packages.agentic.agents.synthesis.node import synthesis_node
from packages.agentic.agents.reviewer.node import reviewer_node
from packages.agentic.llm_client import LLMFailureError, llm_node_context
from packages.knowledge.persistent_memory import PersistentMemoryStore
from packages.agentic.agents.synthesis.diagram import generate_mermaid_architecture
from packages.agentic.agents.synthesis.quickstart import generate_grounded_quickstart

def route_reflection(state: RepoIntelState) -> str:
    # Pure routing: reads the verdict the reviewer produced. The revision
    # budget is owned by reviewer_node, which increments
    # ``reflection_iteration_count`` as a persisted state update and forces
    # verdict=NEEDS_HUMAN_REVIEW once the budget is exhausted — so a
    # REVISE_SYNTHESIS verdict here always still has budget remaining, and
    # the loop is bounded by construction.
    if not state.review_report:
        return "EXIT"

    verdict = state.review_report.verdict
    if verdict == ReviewVerdict.REVISE_SYNTHESIS:
        return "Synthesis"
    return "EXIT"

def build_reflection_subgraph(llm_client, facade):
    subgraph = StateGraph(RepoIntelState)

    async def _synthesis_step(state: RepoIntelState):
        additional_sections = []
        if hasattr(facade, "get_architecture_snapshot") and hasattr(
            facade, "get_dependency_manifest"
        ):
            try:
                snapshot = await facade.get_architecture_snapshot(state.repo_id)
                dependencies = await facade.get_dependency_manifest(state.repo_id)
                diagram_section = generate_mermaid_architecture(snapshot, dependencies)
                if diagram_section:
                    additional_sections.append(diagram_section)
            except Exception:
                pass
        if (
            state.artifact_type.upper().replace(".MD", "") == "README"
            and hasattr(facade, "get_quickstart_facts")
        ):
            try:
                quickstart = generate_grounded_quickstart(
                    await facade.get_quickstart_facts(state.repo_id)
                )
                if quickstart:
                    additional_sections.append(quickstart)
            except Exception:
                pass
        with llm_node_context("Synthesis"):
            return synthesis_node(
                state,
                llm_client,
                additional_sections=additional_sections or None,
            )

    async def _reviewer_step(state: RepoIntelState, runtime: Runtime):
        with llm_node_context("Reviewer"):
            result = await reviewer_node(state, facade, llm_client)
        report = result.get("review_report")
        if runtime.store is not None and report is not None:
            await PersistentMemoryStore(runtime.store).save_review_issues(
                state.repo_id, state.run_id, report.issues
            )
        return result

    subgraph.add_node(
        "Synthesis",
        _synthesis_step,
        retry_policy=RetryPolicy(max_attempts=2, retry_on=(LLMFailureError,)),
    )
    subgraph.add_node(
        "Reviewer",
        _reviewer_step,
        retry_policy=RetryPolicy(max_attempts=2, retry_on=(LLMFailureError,)),
    )

    subgraph.set_entry_point("Synthesis")
    subgraph.add_edge("Synthesis", "Reviewer")

    subgraph.add_conditional_edges(
        "Reviewer",
        route_reflection,
        {
            "Synthesis": "Synthesis",
            "EXIT": END
        }
    )

    return subgraph.compile()
