from packages.agentic.state.schema import ArtifactDraft
from packages.shared.llm_mode import LLMMode


def render_artifact(draft: ArtifactDraft, llm_mode: LLMMode, degraded: bool) -> str:
    """Render grounded sections with machine-readable execution-mode metadata."""
    sections = [
        f"## {section.section_title}\n\n{section.content}\n"
        for section in draft.sections
    ]
    front_matter = (
        "---\n"
        f"llm_mode: {llm_mode}\n"
        f"degraded: {str(degraded).lower()}\n"
        "---\n\n"
    )
    return front_matter + "\n".join(sections)