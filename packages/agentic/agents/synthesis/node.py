import json
from typing import Dict, Any, List
from packages.agentic.state.schema import RepoIntelState, ArtifactDraft, ArtifactSection
from packages.agentic.llm_client import LLMClient

SYNTHESIS_SYSTEM_PROMPT = """
You are a technical documentation synthesis agent.
Your goal is to synthesize raw code evidence into clear, highly accurate, grounded Markdown documentation sections.

STRICT GROUNDING RULES:
1. Every factual claim MUST be grounded in the provided Task Evidence citations.
2. Never invent features, dependencies, flags, or configuration options.
3. If evidence for a requested section is missing or weak, set `insufficient_evidence` to true and explicitly state the limitation in the content.
4. Pass through exact evidence citations to support each section.
"""

def synthesis_node(state: RepoIntelState, llm_client: LLMClient) -> Dict[str, Any]:
    if not state.plan:
        return {"errors": state.errors + ["Synthesis executed without plan"]}

    sections: List[ArtifactSection] = []

    for task in state.plan.tasks:
        evidence = state.evidence_store.get(task.task_id)
        citations_data = [c.model_dump() for c in evidence.citations] if evidence else []

        prompt = f"""
        Target Section: {task.target_section}
        Task Objective: {task.objective}
        Available Evidence Citations:
        {json.dumps(citations_data, indent=2)}
        
        Synthesize this section. Ensure all claims reference the provided evidence citations.
        """

        sec = llm_client.complete_structured(
            prompt=prompt,
            schema=ArtifactSection,
            system=SYNTHESIS_SYSTEM_PROMPT
        )
        sections.append(sec)

    version = (state.artifact_draft.version + 1) if state.artifact_draft else 1
    new_draft = ArtifactDraft(
        version=version,
        artifact_type=state.artifact_type,
        sections=sections
    )

    draft_history = list(state.draft_history)
    if state.artifact_draft:
        draft_history.append(state.artifact_draft)

    return {
        "artifact_draft": new_draft,
        "draft_history": draft_history
    }
