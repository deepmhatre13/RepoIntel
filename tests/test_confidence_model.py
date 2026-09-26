"""Confidence calibration tests (Phase 4).

A direct AST answer to an exact symbol/file question must receive materially
higher confidence than an answer based only on semantic similarity. These tests
exercise ``confidence_from_relevance`` directly.
"""

import uuid
from pathlib import Path

from packages.knowledge.relevance import (
    confidence_from_relevance,
    evaluate_relevance,
    RelevanceLevel,
)
from packages.knowledge.vector_store import RetrievedEvidenceItem

DATASET = Path(__file__).resolve().parent.parent / "evals" / "datasets" / "python_library"
REPO_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, str(DATASET)))


def _ev(file_path, symbol, content):
    return RetrievedEvidenceItem(
        repo_id=REPO_ID, file_path=file_path, symbol_name=symbol,
        line_start=1, line_end=5, content=content,
        retrieval_source="hybrid", relevance_score=0.8,
    )


def test_direct_ast_evidence_yields_high_confidence():
    ev = [_ev("src/calculator.py", "Calculator.add", "def add(self, a, b): return a + b")]
    decision = evaluate_relevance("What functions are defined in calculator.py?", ev)
    assert confidence_from_relevance(decision, len(ev), llm_available=False) == "HIGH"


def test_direct_symbol_lookup_yields_high_confidence():
    ev = [_ev("src/calculator.py", "Calculator", "class Calculator:")]
    decision = evaluate_relevance("What does Calculator do?", ev)
    assert confidence_from_relevance(decision, len(ev), llm_available=False) == "HIGH"


def test_semantic_only_evidence_is_lower_than_direct():
    direct = [_ev("src/calculator.py", "Calculator", "class Calculator:")]
    semantic = [
        _ev("src/calculator.py", "main", "def main(): print(1)"),
        _ev("pyproject.toml", None, "[tool.poetry]\nname='x'"),
    ]
    direct_decision = evaluate_relevance("What does Calculator do?", direct)
    semantic_decision = evaluate_relevance("What is this project about?", semantic)

    direct_conf = confidence_from_relevance(direct_decision, 1, llm_available=True)
    semantic_conf = confidence_from_relevance(semantic_decision, 2, llm_available=True)

    # A direct answer to an exact symbol question outweighs a generic,
    # semantic-only overview question.
    assert direct_conf == "HIGH"
    assert direct_conf != semantic_conf
    assert semantic_conf == "LOW"


def test_no_evidence_yields_low_confidence():
    decision = evaluate_relevance("What PostgreSQL tables?", [_ev("src/calculator.py",
                                                                 "Calculator", "class Calculator:")])
    assert decision.level is RelevanceLevel.NO_RELEVANT_EVIDENCE
    assert confidence_from_relevance(decision, 0, llm_available=True) == "LOW"


def test_weak_evidence_yields_at_most_low_confidence():
    ev = [_ev("src/calculator.py", "Calculator", "class Calculator:")]
    decision = evaluate_relevance("How many total stars in the sky?", ev)
    assert confidence_from_relevance(decision, 1, llm_available=False) == "LOW"