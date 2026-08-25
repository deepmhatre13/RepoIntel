import pytest
from packages.agentic.state.schema import RepoIntelState, ReviewReport, ReviewVerdict
from packages.agentic.graphs.reflection_subgraph import route_reflection
from packages.agentic.graphs.phase1_main_graph import route_main

def test_reflection_routing():
    state = RepoIntelState(
        run_id="test", thread_id="test", repo_id="r1", repo_url="http://example.com", archetype="SERVICE"
    )
    
    # 1. APPROVE -> EXIT reflection subgraph, Finalize main graph
    state.review_report = ReviewReport(verdict=ReviewVerdict.APPROVE)
    assert route_reflection(state) == "EXIT"
    assert route_main(state) == "Finalize"

    # 2. REVISE_SYNTHESIS -> Synthesis reflection subgraph
    state.review_report = ReviewReport(verdict=ReviewVerdict.REVISE_SYNTHESIS)
    assert route_reflection(state) == "Synthesis"

    # 3. REVISE_PLAN -> Planner main graph
    state.review_report = ReviewReport(verdict=ReviewVerdict.REVISE_PLAN)
    assert route_main(state) == "Planner"

    # 4. NEEDS_HUMAN_REVIEW -> HumanReview main graph
    state.review_report = ReviewReport(verdict=ReviewVerdict.NEEDS_HUMAN_REVIEW)
    assert route_main(state) == "HumanReview"
