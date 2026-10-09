from packages.agentic.artifact_rendering import render_artifact
from packages.agentic.artifact_engine import ArtifactEngine, GeneratedArtifact
from packages.agentic.state.schema import ArtifactDraft, ArtifactSection


def test_artifact_rendering_includes_llm_mode_front_matter():
    content = render_artifact(
        ArtifactDraft(version=1, sections=[ArtifactSection(
            section_title="Overview",
            content="Evidence-grounded summary.",
        )]),
        llm_mode="DEGRADED_FALLBACK",
        degraded=True,
    )

    assert content.startswith(
        "---\nllm_mode: DEGRADED_FALLBACK\ndegraded: true\n---\n\n"
    )
    assert "## Overview\n\nEvidence-grounded summary." in content


def test_artifact_engine_rendering_includes_llm_mode_front_matter():
    artifact = GeneratedArtifact(
        artifact_type="README",
        repository_id="repo-1",
        title="README",
        degraded=True,
        llm_mode="DEGRADED_FALLBACK",
    )

    content = ArtifactEngine._render(artifact)

    assert content.startswith(
        "---\nllm_mode: DEGRADED_FALLBACK\ndegraded: true\n---\n\n"
    )