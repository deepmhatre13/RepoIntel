from __future__ import annotations

from typing import Any

from packages.agentic.llm_client import LLMClient, llm_node_context
from packages.agentic.state.schema import Plan, ReviewReport, ReviewVerdict


class _LocalStructuredResponse:
    def __init__(self, data: Any):
        self.data = data


def test_llm_client_reports_local_tier(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen2.5-coder:7b-instruct")
    monkeypatch.setattr(LLMClient, "_probe_gemini", lambda self: False)
    monkeypatch.setattr(LLMClient, "_probe_local_llm", lambda self: True)

    client = LLMClient()

    assert client.get_mode() == "LOCAL_OSS_LLM"
    assert client.is_available() is True


def test_complete_structured_falls_back_to_local_llm(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen2.5-coder:7b-instruct")
    monkeypatch.setattr(LLMClient, "_probe_gemini", lambda self: False)
    monkeypatch.setattr(LLMClient, "_probe_local_llm", lambda self: True)

    calls: list[dict[str, Any]] = []

    def fake_local_call(self, prompt: str, schema: Any, system: str | None = None) -> Any:
        calls.append({"prompt": prompt, "schema": schema.__name__, "system": system})
        if schema is Plan:
            return Plan(plan_id="plan-local", repo_id="repo-1", artifact_type="README", tasks=[])
        if schema is ReviewReport:
            return ReviewReport(verdict=ReviewVerdict.APPROVE, issues=[], groundedness_score=1.0, coverage_score=1.0)
        return schema()

    monkeypatch.setattr(LLMClient, "_call_local_structured_model", fake_local_call)

    client = LLMClient()
    result = client.complete_structured("Need a plan", schema=Plan, system="sys")

    assert isinstance(result, Plan)
    assert result.plan_id == "plan-local"
    assert client.get_mode() == "LOCAL_OSS_LLM"
    assert calls


def test_gemini_structured_completion_uses_streaming_runnable(monkeypatch):
    expected = Plan(
        plan_id="plan-streamed",
        repo_id="repo-1",
        artifact_type="README",
        tasks=[],
    )
    calls = {"stream": 0, "invoke": 0}

    class StructuredRunnable:
        def stream(self, messages, config=None):
            calls["stream"] += 1
            assert config["metadata"]["llm_mode"] == "LIVE_GEMINI"
            assert config["metadata"]["node"] == "unattributed"
            yield expected

        def invoke(self, messages):
            calls["invoke"] += 1
            raise AssertionError("structured runnable must stream")

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            assert schema is Plan
            return StructuredRunnable()

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        FakeChatGoogleGenerativeAI,
    )
    monkeypatch.setattr(LLMClient, "_probe_local_llm", lambda self: False)
    client = LLMClient(api_key="test-key")

    result = client.complete_structured("Make a README plan", Plan)

    assert result == expected
    assert calls == {"stream": 1, "invoke": 0}
    assert client.last_mode == "LIVE_GEMINI"


def test_usage_summary_groups_calls_by_node_and_model_tier():
    client = LLMClient(api_key="")
    client.last_mode = "LOCAL_OSS_LLM"

    with llm_node_context("Retriever"):
        client._record_usage(12.5, 30, 12)

    summary = client.get_usage_summary()

    assert summary["calls"] == 1
    assert summary["calls_by_node"]["Retriever"] == {
        "calls": 1,
        "tokens_in": 30,
        "tokens_out": 12,
        "duration_ms": 12.5,
        "estimated_cost_usd": 0.0,
        "llm_modes": ["LOCAL_OSS_LLM"],
    }
    assert summary["calls_detail"][0]["model"] == client.local_llm_model
