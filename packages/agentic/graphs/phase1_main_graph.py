from langgraph.graph import StateGraph, END
from packages.agentic.state.schema import RepoIntelState, ReviewVerdict
from packages.agentic.agents.planner.node import planner_node
from packages.agentic.agents.retriever.node import retriever_node
from packages.agentic.graphs.reflection_subgraph import build_reflection_subgraph

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

def build_main_graph(llm_client, facade):
    graph = StateGraph(RepoIntelState)

    def _planner_step(state: RepoIntelState):
        return planner_node(state, llm_client)

    async def _retriever_step(state: RepoIntelState):
        return await retriever_node(state, facade, llm_client)

    reflection_subgraph = build_reflection_subgraph(llm_client, facade)

    def _finalize_step(state: RepoIntelState):
        return {"route_decision": "FINALIZED"}

    def _human_review_step(state: RepoIntelState):
        if state.human_feedback:
            if state.human_feedback.decision == "APPROVE":
                return {"route_decision": "FINALIZED"}
            elif state.human_feedback.decision == "REQUEST_REVISION":
                return {"route_decision": "REVISE_SYNTHESIS"}
        return {"route_decision": "NEEDS_HUMAN_REVIEW"}

    graph.add_node("Planner", _planner_step)
    graph.add_node("Retriever", _retriever_step)
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

    return graph.compile()
