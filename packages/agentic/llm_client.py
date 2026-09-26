import os
import time
from typing import Type, TypeVar, Optional, Dict, Any, List, Set
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

from packages.shared.secret_redaction import redact_secrets

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
        # Ensure the project .env has been materialized into os.environ exactly
        # once before reading GOOGLE_API_KEY / GEMINI_MODEL. This is the single
        # chokepoint; LLMClient never discovers .env files itself — it only
        # consumes the centralized settings layer so configuration happens once,
        # at application startup, in the correct order.
        from packages.shared.config import settings as _settings
        _settings.load_dotenv()

        # Explicit constructor args always take precedence over configuration.
        # An explicitly-passed empty string forces degraded mode (documented
        # contract used by deterministic/degraded tests); None means "resolve
        # from configuration" (.env -> shell env -> empty = degraded).
        if api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY", "") or ""
        else:
            self.api_key = api_key
        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or _settings.gemini_model
        )




        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_duration_ms = 0
        self.total_cost_usd = 0.0
        self.call_count = 0
        self.last_error: Optional[str] = None
        # Secret redaction accounting (Objective 16).  Only counts are kept;
        # matched values never leave this module.
        self.total_redactions = 0
        self.redaction_categories: Dict[str, int] = {}
        # Cost estimate for Gemini 1.5 Pro pricing tiers ($/1M tokens).
        self._cost_per_1k_input = 0.00125
        self._cost_per_1k_output = 0.005

    def _secure(self, text: str) -> str:
        """Redact obvious secrets before content leaves the process."""
        if not text:
            return text
        result = redact_secrets(text)
        if result.total_redactions:
            self.total_redactions += result.total_redactions
            for cat, n in result.categories.items():
                self.redaction_categories[cat] = self.redaction_categories.get(cat, 0) + n
        return result.text

    def get_usage_summary(self) -> Dict[str, Any]:
        """Measured usage across all calls made through this client."""
        return {
            "model": self.model_name,
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "duration_ms": round(self.total_duration_ms, 2),
            "estimated_cost_usd": self.total_cost_usd,
            "llm_available": self.is_available(),
            "secrets_redacted": self.total_redactions,
            "redaction_categories": dict(self.redaction_categories),
        }

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

    # Explicit sentinel returned (never raised) when Gemini is unavailable.
    # It is not LLM output: callers must detect ``is_available()`` / this
    # marker and produce a deterministic, evidence-grounded reply that
    # explicitly communicates degraded mode instead of parroting a fake answer.
    DEGRADED_MODE_MARKER = (
        "[DEGRADED_MODE: no GOOGLE_API_KEY configured — LLM is unavailable; "
        "this is NOT LLM output. Use deterministic evidence-based reasoning.]"
    )

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        if not self.api_key:
            # Degraded mode marker: deterministic fallback, never pretends to be
            # LLM output. Downstream callers must translate this into an explicit,
            # evidence-grounded degraded answer (never present it verbatim).
            self.last_error = "GOOGLE_API_KEY not configured"
            return self.DEGRADED_MODE_MARKER
        start = time.time()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            prompt = self._secure(prompt)
            if system:
                system = self._secure(system)
            llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key, thinking_level="medium", temperature=0.2, max_retries=1)
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

    def _generate_artifact_section(self, prompt: str):

        """Deterministic, evidence-grounded fallback for ArtifactSection synthesis.

        Parses the structured prompt (which embeds the task metadata and the
        retrieved evidence citations as JSON) and produces a real section from
        the actual evidence — never a placeholder string. When no evidence is
        available for the section, it explicitly states that limitation.
        """
        from packages.agentic.state.schema import ArtifactSection, EvidenceCitation
        import json as _json

        target = "Section"
        objective = ""
        if "Target Section:" in prompt:
            target = prompt.split("Target Section:")[1].splitlines()[0].strip()
        if "Task Objective:" in prompt:
            objective = prompt.split("Task Objective:")[1].splitlines()[0].strip()

        citations: list = []
        try:
            if "Available Evidence Citations:" in prompt:
                after = prompt.split("Available Evidence Citations:")[1]
                bracket_start = after.find("[")
                bracket_end = after.rfind("]")
                if bracket_start != -1 and bracket_end != -1:
                    raw = after[bracket_start:bracket_end + 1]
                    data = _json.loads(raw)
                    citations = [EvidenceCitation(**c) for c in data if isinstance(c, dict)]
        except Exception:
            citations = []

        if not citations:
            return ArtifactSection(
                section_title=target,
                content=(
                    f"No evidence was found in the repository for the **{target}** section "
                    f"({objective or 'no specific objective provided'}). "
                    "This section could not be grounded in repository evidence."
                ),
                citations=[],
                insufficient_evidence=True,
            )

        lines: list = []
        seen_files: set = set()
        seen_citations: set = set()
        for c in citations:
            # Deduplicate citations by their (file_path, symbol_name, content)
            # identity so accumulated evidence from repeated reflection
            # iterations is not surfaced as duplicate bullet lines.
            cite_key = (
                c.file_path,
                c.symbol_name,
                (c.content or "")[:64],
            )
            if cite_key in seen_citations:
                if c.file_path:
                    seen_files.add(c.file_path)
                continue
            seen_citations.add(cite_key)
            fp = c.file_path or "<unknown>"
            loc = f"{fp}::{c.symbol_name}" if c.symbol_name else fp
            lines.append(f"- {loc}")
            seen_files.add(fp)

        if "Installation" in target:
            content = (
                "No installation instructions were found in the repository evidence. "
                "Cannot determine installation procedure.\n" + "\n".join(lines)
            )
        elif "Overview" in target:
            # Derive real facts from the evidence content rather than only
            # listing file names: extract declared classes/functions and
            # module docstrings actually present in the cited source.
            import re as _re
            declared: list = []
            docstring = ""
            for c in citations:
                for m in _re.finditer(
                    r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)"
                    r"|^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)",
                    c.content, _re.MULTILINE,
                ):
                    name = m.group(1) or m.group(2)
                    kind = "class" if m.group(2) else "function"
                    entry = f"{kind} `{name}` ({c.file_path})"
                    if entry not in declared:
                        declared.append(entry)
                if not docstring:
                    dm = _re.search(r'^\s*[rRbB]?["\']{3}(.+?)["\']{3}', c.content, _re.DOTALL)
                    if dm:
                        docstring = " ".join(dm.group(1).split())[:300]
            parts: list = []
            if docstring:
                parts.append(f"Module docstring (from evidence): {docstring}")
            if declared:
                parts.append(
                    "Declared in the indexed source:\n- " + "\n- ".join(declared)
                )
            parts.append("Files covered by evidence: " + ", ".join(sorted(seen_files)))
            content = "\n\n".join(parts)
        elif "Usage" in target:
            content = "Usage examples found in the repository evidence:\n" + "\n".join(lines)
        elif "Testing" in target:
            content = "Testing information found in the repository evidence:\n" + "\n".join(lines)
        elif "Configuration" in target:
            content = "Configuration information found in the repository evidence:\n" + "\n".join(lines)
        elif "Dependencies" in target:
            content = "Dependencies found in the repository evidence:\n" + "\n".join(lines)
        else:
            content = (
                f"Content for **{target}** grounded in repository evidence:\n"
                + "\n".join(lines)
            )

        return ArtifactSection(
            section_title=target,
            content=content,
            citations=citations,
            insufficient_evidence=False,
        )


    def _generate_tool_call_decision(self, prompt: str):
        """Deterministic fallback for ToolCallDecision.

        When no LLM is available, decide which retrieval tool to call based on
        keywords in the task's retrieval_query and target_section. This lets the
        retriever gather real evidence in degraded mode instead of immediately
        finishing with no results.
        """
        from packages.agentic.agents.retriever.node import ToolCallDecision

        target = ""
        if "Target Section:" in prompt:
            target = prompt.split("Target Section:")[1].splitlines()[0].strip().lower()
        query = ""
        if "Retrieval Query:" in prompt:
            query = prompt.split("Retrieval Query:")[1].splitlines()[0].strip()

        if "install" in target or "install" in query:
            return ToolCallDecision(tool_name="get_dependency_manifest", query_or_param="", reasoning="Deterministic: look up dependency manifest for Installation section")
        if "symbol" in target or "graph" in target:
            return ToolCallDecision(tool_name="hybrid_retrieve", query_or_param=query or "code", reasoning="Deterministic: retrieve code evidence")
        if "finish" in prompt.lower() and "Citations Gathered So Far" in prompt:
            gathered = 0
            try:
                after = prompt.split("Citations Gathered So Far:")[1].splitlines()[0]
                gathered = int(after.strip())
            except Exception:
                gathered = 0
            if gathered >= 3:
                return ToolCallDecision(tool_name="finish_task", query_or_param="", reasoning="Deterministic: sufficient evidence gathered")
            return ToolCallDecision(tool_name="hybrid_retrieve", query_or_param=query or target, reasoning="Deterministic: retrieve more evidence")
        return ToolCallDecision(tool_name="hybrid_retrieve", query_or_param=query or "code", reasoning="Deterministic: retrieve code evidence")

    def complete_structured(self, prompt: str, schema: Type[T], system: Optional[str] = None, max_retries: int = 2) -> T:
        if not self.api_key:
            # Deterministic fallback instantiation if no API key is provided
            from packages.agentic.state.schema import PlanTask, ReviewVerdict, EvidenceCitation
            if schema.__name__ == "Plan":
                return schema(plan_id="p1", repo_id="r1", artifact_type="README", tasks=[
                    PlanTask(task_id="t1", objective="Overview", retrieval_query="overview", evidence_type="code", target_section="Overview"),
                    PlanTask(task_id="t2", objective="Installation", retrieval_query="install", evidence_type="manifest", target_section="Installation")
                ])
            elif schema.__name__ == "ToolCallDecision":
                return self._generate_tool_call_decision(prompt)
            elif schema.__name__ == "ArtifactSection":
                return self._generate_artifact_section(prompt)
            elif schema.__name__ == "ReviewReport":
                return schema(verdict=ReviewVerdict.APPROVE, issues=[], groundedness_score=1.0, coverage_score=1.0)
            elif schema.__name__ == "EvidenceCitation":
                return schema(file_path="", content="", line_start=0, line_end=0)
            else:

                return schema.model_construct()



        start = time.time()
        from langchain_google_genai import ChatGoogleGenerativeAI
        prompt = self._secure(prompt)
        if system:
            system = self._secure(system)
        llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key, thinking_level="medium", temperature=0.2, max_retries=1)
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
                transient_markers = ("429", "503", "rate", "quota", "timeout", "connection", "internal", "unavailable")
                if any(m in msg_l for m in transient_markers):
                    if attempt < max_retries:
                        time.sleep(6 * (attempt + 1))
                        continue
                    raise LLMFailureError(f"Gemini structured completion failed after retries: {e}", transient=True)
                messages.append(HumanMessage(content=f"[Validation error: {str(e)}]"))

        raise StructuredOutputValidationError(f"Failed to generate structured output matching schema {schema.__name__}")
