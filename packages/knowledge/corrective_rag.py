from enum import Enum

from packages.agentic.state.schema import PlanTask, TaskEvidence


class EvidenceSufficiency(str, Enum):
    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"


def assess_evidence_sufficiency(
    evidence: TaskEvidence,
    task: PlanTask,
) -> EvidenceSufficiency:
    if not evidence.citations:
        return EvidenceSufficiency.INCORRECT

    best_score = max(citation.relevance_score for citation in evidence.citations)
    if best_score >= 0.45:
        return EvidenceSufficiency.CORRECT
    if best_score >= 0.15:
        return EvidenceSufficiency.AMBIGUOUS
    return EvidenceSufficiency.INCORRECT