"""Regression tests for the retrieval relevance gate (Phase 3).

A question must not be considered answered merely because top-k documents were
retrieved. These tests verify, against the python_library fixture, that:

* strongly supported questions are HIGH_RELEVANCE
* weakly related questions are LOW/MEDIUM_RELEVANCE
* unsupported technology questions (PostgreSQL/Kubernetes/Redis/Kafka/Lambda)
  are classified NO_RELEVANT_EVIDENCE and rejected by the QA path
* completely unrelated questions are NO_RELEVANT_EVIDENCE
* direct symbol / file lookups are classified HIGH_RELEVANCE
"""

import uuid
from pathlib import Path

import pytest
from packages.knowledge.relevance import (
    evaluate_relevance,
    RelevanceLevel,
    significant_terms,
)
from packages.knowledge.vector_store import RetrievedEvidenceItem

DATASET = (
    Path(__file__).resolve().parent.parent
    / "evals" / "datasets" / "python_library"
)

REPO_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, str(DATASET)))

CALC_CHUNK = (
    "class Calculator:\n"
    "    def add(self, a, b):\n"
    "        return a + b\n"
    "    def multiply(self, a, b):\n"
    "        return a * b\n"
)


def _ev(file_path, symbol, content, score=0.8):
    return RetrievedEvidenceItem(
        repo_id=REPO_ID,
        file_path=file_path,
        symbol_name=symbol,
        line_start=1,
        line_end=10,
        content=content,
        retrieval_source="hybrid",
        relevance_score=score,
    )


def _calculator_evidence():
    """Evidence mirroring what the python_library fixture indexes."""
    return [
        _ev("src/calculator.py", "Calculator.add",
            "def add(self, a, b): return a + b"),
        _ev("src/calculator.py", "Calculator.multiply",
            "def multiply(self, a, b): return a * b"),
        _ev("src/calculator.py", "main",
            "def main():  print('github calculators')"),
        _ev("pyproject.toml", None,
            "[tool.poetry]\nname = 'python-library'"),
    ]


def test_strongly_supported_question_is_high():
    decision = evaluate_relevance(
        "What functions are defined in calculator.py?", _calculator_evidence()
    )
    assert decision.level is RelevanceLevel.HIGH_RELEVANCE


def test_direct_symbol_lookup_is_high():
    decision = evaluate_relevance(
        "What does Calculator.add do?", _calculator_evidence()
    )
    assert decision.level is RelevanceLevel.HIGH_RELEVANCE


def test_filename_specific_lookup_is_high():
    decision = evaluate_relevance("calculator.py", _calculator_evidence())
    assert decision.level is RelevanceLevel.HIGH_RELEVANCE


def test_unsupported_technology_postgresql_is_rejected():
    # PostgreSQL is absent from the fixture; calculator evidence must NOT be
    # treated as relevant to a PostgreSQL question.
    decision = evaluate_relevance(
        "What PostgreSQL tables does this repository use?", _calculator_evidence()
    )
    assert decision.level is RelevanceLevel.NO_RELEVANT_EVIDENCE


def test_unsupported_technology_kubernetes_is_rejected():
    decision = evaluate_relevance(
        "What Kubernetes controllers does this repository implement?",
        _calculator_evidence(),
    )
    assert decision.level is RelevanceLevel.NO_RELEVANT_EVIDENCE


def test_unsupported_technology_redis_cache_keys_is_rejected():
    decision = evaluate_relevance(
        "What Redis cache keys are used?", _calculator_evidence()
    )
    assert decision.level is RelevanceLevel.NO_RELEVANT_EVIDENCE


def test_unsupported_technology_aws_lambda_is_rejected():
    decision = evaluate_relevance(
        "Which AWS Lambda functions exist?", _calculator_evidence()
    )
    assert decision.level is RelevanceLevel.NO_RELEVANT_EVIDENCE


def test_unsupported_technology_kafka_topics_is_rejected():
    decision = evaluate_relevance(
        "What Kafka topics are published?", _calculator_evidence()
    )
    assert decision.level is RelevanceLevel.NO_RELEVANT_EVIDENCE


def test_completely_unrelated_question_is_rejected():
    decision = evaluate_relevance(
        "What is the capital of France?", _calculator_evidence()
    )
    assert decision.level is RelevanceLevel.NO_RELEVANT_EVIDENCE


def test_weakly_related_question_is_low_or_medium():
    # "who" / vague probing maps to family of functions/inputs; only generic
    # overlap => LOW or MEDIUM.
    decision = evaluate_relevance(
        "What does the library use print for?", _calculator_evidence()
    )
    assert decision.level in (
        RelevanceLevel.LOW_RELEVANCE,
        RelevanceLevel.MEDIUM_RELEVANCE,
    )


def test_significant_terms_are_stopword_filtered():
    assert "repository" not in significant_terms("what does this repository do")
    assert "use" not in significant_terms("Why does this use X")
    assert "calculator" in significant_terms("what functions in calculator")
    assert "postgresql" in significant_terms(
        "What PostgreSQL tables does this repository use?"
    )