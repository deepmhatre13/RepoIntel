import pytest

from packages.agentic.agents.reviewer.node import (
    ConsistencyReport,
    reviewer_node,
)
from packages.agentic.state.schema import (
    ArtifactDraft,
    ArtifactSection,
    RepoIntelState,
    ReviewReport,
    ReviewVerdict,
)


class ConsistencyCheckingLLM:
    def __init__(self, inconsistencies):
        self.inconsistencies = inconsistencies
        self.consistency_prompt = ""

    def complete_structured(self, prompt, schema, system=None):
        if schema is ReviewReport:
            return ReviewReport(verdict=ReviewVerdict.APPROVE)
        assert schema is ConsistencyReport
        self.consistency_prompt = prompt
        return ConsistencyReport(inconsistencies=self.inconsistencies)


class EmptyFacade:
    pass


@pytest.mark.asyncio
async def test_reviewer_flags_cross_section_configuration_contradiction():
    state = RepoIntelState(
        run_id="run-consistency",
        thread_id="thread-consistency",
        repo_id="repo-consistency",
        repo_url="https://example.com/repo",
        archetype="SERVICE",
        artifact_draft=ArtifactDraft(
            version=1,
            sections=[
                ArtifactSection(
                    section_title="Configuration",
                    content="The service requires DATABASE_URL.",
                ),
                ArtifactSection(
                    section_title="Docker",
                    content="The container does not set DATABASE_URL.",
                ),
                ArtifactSection(
                    section_title="Architecture Diagram",
                    content="```mermaid\nflowchart TD\n```",
                ),
            ],
        ),
    )
    llm = ConsistencyCheckingLLM([
        "Configuration requires DATABASE_URL, but Docker does not set it."
    ])

    result = await reviewer_node(state, EmptyFacade(), llm)

    assert "Configuration" in llm.consistency_prompt
    assert "Docker" in llm.consistency_prompt
    assert "Architecture Diagram" in llm.consistency_prompt
    assert result["review_report"].verdict == ReviewVerdict.REVISE_SYNTHESIS
    assert any(
        issue.severity == "HIGH"
        and "DATABASE_URL" in issue.description
        for issue in result["review_report"].issues
    )


@pytest.mark.asyncio
async def test_consistent_sections_keep_approval():
    state = RepoIntelState(
        run_id="run-consistent",
        thread_id="thread-consistent",
        repo_id="repo-consistent",
        repo_url="https://example.com/repo",
        archetype="SERVICE",
        artifact_draft=ArtifactDraft(
            version=1,
            sections=[
                ArtifactSection(section_title="Usage", content="Run the service."),
                ArtifactSection(section_title="Docker", content="Docker runs the service."),
            ],
        ),
    )

    result = await reviewer_node(
        state, EmptyFacade(), ConsistencyCheckingLLM([])
    )

    assert result["review_report"].verdict == ReviewVerdict.APPROVE