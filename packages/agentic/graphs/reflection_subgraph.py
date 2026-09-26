from langgraph.graph import StateGraph, END
from packages.agentic.state.schema import RepoIntelState, ReviewVerdict
from packages.agentic.agents.synthesis.node import synthesis_node
from packages.agentic.agents.reviewer.node import reviewer_node

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

    def _synthesis_step(state: RepoIntelState):
        return synthesis_node(state, llm_client)

    async def _reviewer_step(state: RepoIntelState):
        return await reviewer_node(state, facade, llm_client)

    subgraph.add_node("Synthesis", _synthesis_step)
    subgraph.add_node("Reviewer", _reviewer_step)

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
