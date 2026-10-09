from __future__ import annotations

from packages.agentic.state.schema import (
    ArtifactSection,
    EvidenceCitation,
    Plan,
    PlanTask,
    RepoIntelState,
    TaskEvidence,
)
from packages.agentic.agents.synthesis.node import synthesis_node


class DummyLLMClient:
    def __init__(self):
        self.calls = []

    def complete_structured(self, prompt, schema, system=None):
        self.calls.append({
            "target": prompt.split("Target Section:", 1)[1].splitlines()[0].strip(),
            "prompt": prompt,
        })
        return ArtifactSection(
            section_title=prompt.split("Target Section:", 1)[1].splitlines()[0].strip(),
            content=f"Section: {prompt.split('Target Section: ', 1)[1].splitlines()[0].strip()}",
            citations=[],
            insufficient_evidence=False,
        )


def test_send_fanout_synthesis_dispatches_all_tasks():
    llm = DummyLLMClient()
    state = RepoIntelState(
        run_id="run-1",
        thread_id="thread-1",
        repo_id="repo-1",
        repo_url="https://example.com/repo",
        archetype="SERVICE",
        plan=Plan(
            plan_id="plan-1",
            repo_id="repo-1",
            artifact_type="README",
            document_outline=["Overview", "Usage", "Testing"],
            tasks=[
                PlanTask(task_id="t1", objective="First", retrieval_query="first", evidence_type="code", target_section="Overview", priority=1),
                PlanTask(task_id="t2", objective="Second", retrieval_query="second", evidence_type="code", target_section="Usage", priority=2),
                PlanTask(task_id="t3", objective="Third", retrieval_query="third", evidence_type="code", target_section="Testing", priority=3),
            ],
        ),
        evidence_store={
            "t1": TaskEvidence(task_id="t1", query="first", citations=[]),
            "t2": TaskEvidence(task_id="t2", query="second", citations=[]),
            "t3": TaskEvidence(task_id="t3", query="third", citations=[]),
        },
    )

    result = synthesis_node(state, llm)

    assert len(llm.calls) == 4
    assert len(result["artifact_draft"].sections) == 4
    assert llm.calls[-1]["target"] == "Executive Summary"
    assert "Section: Overview" in llm.calls[-1]["prompt"]
    assert result["artifact_draft"].sections[0].section_title == "Executive Summary"
    titles = [section.section_title for section in result["artifact_draft"].sections]
    assert titles == ["Executive Summary", "Overview", "Usage", "Testing"]


def test_send_fanout_synthesis_sorts_results_by_document_outline():
    llm = DummyLLMClient()
    state = RepoIntelState(
        run_id="run-2",
        thread_id="thread-2",
        repo_id="repo-2",
        repo_url="https://example.com/repo",
        archetype="SERVICE",
        plan=Plan(
            plan_id="plan-2",
            repo_id="repo-2",
            artifact_type="README",
            document_outline=["Testing", "Overview", "Usage"],
            tasks=[
                PlanTask(task_id="t1", objective="First", retrieval_query="first", evidence_type="code", target_section="Overview", priority=1),
                PlanTask(task_id="t2", objective="Second", retrieval_query="second", evidence_type="code", target_section="Usage", priority=2),
                PlanTask(task_id="t3", objective="Third", retrieval_query="third", evidence_type="code", target_section="Testing", priority=3),
            ],
        ),
    )

    diagram = ArtifactSection(
        section_title="Architecture Diagram",
        content="```mermaid\nflowchart TD\n```",
        citations=[EvidenceCitation(
            file_path="src/app.py",
            line_start=1,
            line_end=4,
            content="module import graph",
        )],
    )
    result = synthesis_node(state, llm, additional_sections=[diagram])

    summary_prompt = next(
        call["prompt"] for call in llm.calls
        if call["target"] == "Executive Summary"
    )
    assert "Architecture Diagram" in summary_prompt
    assert "flowchart TD" in summary_prompt
    assert "module import graph" in summary_prompt
    assert [section.section_title for section in result["artifact_draft"].sections] == [
        "Executive Summary", "Testing", "Overview", "Architecture Diagram", "Usage"
    ]
