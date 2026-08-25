import os
import json
import time
import random
from typing import Type, TypeVar, Optional, Dict, Any
from pydantic import BaseModel, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage

T = TypeVar("T", bound=BaseModel)

class StructuredOutputValidationError(Exception):
    pass

class LLMFailureError(Exception):
    """Raised when Gemini is unavailable or fails after retries.

    Category LLM_FAILURE per Phase 43.1 -- retryable when transient.
    """

    def __init__(self, message: str, transient: bool = True):
        super().__init__(message)
        self.transient = transient

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_duration_ms = 0
        self.total_cost_usd = 0.0
        self.call_count = 0
        self.last_error: Optional[str] = None
        # Cost estimate for Gemini 1.5 Pro pricing tiers ($/1M tokens).
        self._cost_per_1k_input = 0.00125
        self._cost_per_1k_output = 0.005

    def is_available(self) -> bool:
        """True when real Gemini calls are possible (API key configured)."""
        return bool(self.api_key)

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return round(
            (input_tokens / 1000.0) * self._cost_per_1k_input
            + (output_tokens / 1000.0) * self._cost_per_1k_output,
            6,
        )

    def _record_usage(self, duration_ms: float, input_tokens: int, output_tokens: int):
        self.total_duration_ms += duration_ms
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd = round(self.total_cost_usd + self._estimate_cost(input_tokens, output_tokens), 6)
        self.call_count += 1

    @staticmethod
    def _approx_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        if not self.api_key:
            # Degraded mode marker: deterministic fallback, never pretends to be LLM output.
            self.last_error = "GOOGLE_API_KEY not configured"
            return "[LLMClient standard completion mock output]"
        start = time.time()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key, temperature=0.2)
            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))
            out = str(llm.invoke(messages).content)
            self._record_usage(
                (time.time() - start) * 1000,
                self._approx_tokens(prompt),
                self._approx_tokens(out),
            )
            return out
        except Exception as e:
            self.last_error = str(e)
            raise LLMFailureError(f"Gemini completion failed: {e}", transient=True)

    def complete_structured(self, prompt: str, schema: Type[T], system: Optional[str] = None, max_retries: int = 2) -> T:
        if not self.api_key:
            # Deterministic fallback instantiation if no API key is provided
            from packages.agentic.state.schema import Plan, PlanTask, ArtifactSection, ReviewReport, ReviewVerdict
            from packages.agentic.agents.retriever.node import ToolCallDecision
            if schema.__name__ == "Plan":
                return schema(plan_id="p1", repo_id="r1", artifact_type="README", tasks=[
                    PlanTask(task_id="t1", objective="Overview", retrieval_query="overview", evidence_type="code", target_section="Overview"),
                    PlanTask(task_id="t2", objective="Installation", retrieval_query="install", evidence_type="manifest", target_section="Installation")
                ])
            elif schema.__name__ == "ToolCallDecision":
                return schema(tool_name="finish_task", query_or_param="", reasoning="Completed analysis")
            elif schema.__name__ == "ArtifactSection":
                target = "Section"
                if "Target Section:" in prompt:
                    target = prompt.split("Target Section:")[1].splitlines()[0].strip()
                return schema(section_title=target, content=f"Content for {target} grounded in code evidence.", citations=[])
            elif schema.__name__ == "ReviewReport":
                return schema(verdict=ReviewVerdict.APPROVE, issues=[], groundedness_score=1.0, coverage_score=1.0)
            else:
                return schema.model_construct()

        start = time.time()
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key, temperature=0.2)
        structured_llm = llm.with_structured_output(schema)
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        for attempt in range(max_retries + 1):
            try:
                res = structured_llm.invoke(messages)
                self._record_usage(
                    (time.time() - start) * 1000,
                    self._approx_tokens(prompt),
                    self._approx_tokens(str(res)),
                )
                if isinstance(res, schema):
                    return res
                elif isinstance(res, dict):
                    return schema.model_validate(res)
            except Exception as e:
                self.last_error = str(e)
                msg_l = str(e).lower()
                transient_markers = ("429", "503", "rate", "quota", "timeout", "connection", "internal")
                if any(m in msg_l for m in transient_markers):
                    raise LLMFailureError(f"Gemini structured completion failed: {e}", transient=True)
                messages.append(HumanMessage(content=f"[Validation error: {str(e)}]"))

        raise StructuredOutputValidationError(f"Failed to generate structured output matching schema {schema.__name__}")
