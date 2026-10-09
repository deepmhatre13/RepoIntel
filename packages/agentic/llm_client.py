import os
import time
import json
import re
import urllib.request
import urllib.error
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Type, TypeVar, Optional, Dict, Any, List, Set
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

from packages.shared.secret_redaction import redact_secrets
from packages.shared.langfuse_tracing import langfuse_generation

T = TypeVar("T", bound=BaseModel)
_CURRENT_LLM_NODE: ContextVar[str] = ContextVar("repointel_llm_node", default="unattributed")


@contextmanager
def llm_node_context(node_name: str):
    token = _CURRENT_LLM_NODE.set(node_name)
    try:
        yield
    finally:
        _CURRENT_LLM_NODE.reset(token)

class StructuredOutputValidationError(Exception):
    pass

class LLMFailureError(Exception):
    """Raised when Gemini is unavailable or fails after retries.

    Category LLM_FAILURE per Phase 43.1 -- retryable when transient.
    """

    def __init__(self, message: str, transient: bool = True):
        super().__init__(message)
        self.transient = transient


def should_run_degraded_artifact_generation() -> bool:
    """Whether artifact generation may proceed without live Gemini.

    Production currently skips artifacts when Gemini is unavailable. Keep this
    policy shared by API and CLI entry points so the behavior changes together.
    """
    return False


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
        self.local_llm_base_url = (
            os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434").rstrip("/")
        )
        self.local_llm_model = (
            os.getenv("LOCAL_LLM_MODEL", "qwen2.5-coder:7b-instruct")
        )
        self.last_mode = "DETERMINISTIC_FALLBACK"

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_duration_ms = 0
        self.total_cost_usd = 0.0
        self.call_count = 0
        self._usage_calls: List[Dict[str, Any]] = []
        self.last_error: Optional[str] = None
        # Secret redaction accounting (Objective 16).  Only counts are kept;
        # matched values never leave this module.
        self.total_redactions = 0
        self.redaction_categories: Dict[str, int] = {}
        # Cost estimate for Gemini 1.5 Pro pricing tiers ($/1M tokens).
        self._cost_per_1k_input = 0.00125
        self._cost_per_1k_output = 0.005

    def get_mode(self) -> str:
        """Return the currently active tier, probing once per client instance."""
        if self._probe_gemini():
            self.last_mode = "LIVE_GEMINI"
        elif self._probe_local_llm():
            self.last_mode = "LOCAL_OSS_LLM"
        else:
            self.last_mode = "DETERMINISTIC_FALLBACK"
        return self.last_mode

    def _probe_gemini(self) -> bool:
        return bool(self.api_key)

    def _probe_local_llm(self) -> bool:
        url = f"{self.local_llm_base_url}/api/tags"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace") or "{}")
            models = payload.get("models", []) if isinstance(payload, dict) else []
            if not models:
                return True
            names = []
            for model in models:
                if isinstance(model, dict):
                    names.append(str(model.get("name", "")))
                else:
                    names.append(str(model))
            preferred = self.local_llm_model
            if not names:
                return True
            normalized = {n.split(":", 1)[0] for n in names}
            return preferred.split(":", 1)[0] in normalized or any(
                preferred in n or n.endswith(f":{preferred}")
                for n in names
            )
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            return False

    def _call_local_structured_model(self, prompt: str, schema: Type[T], system: Optional[str] = None) -> T:
        start = time.time()
        prompt = self._secure(prompt)
        system = self._secure(system) if system else None
        payload = {
            "model": self.local_llm_model,
            "stream": False,
            "messages": [],
            "format": schema.model_json_schema(),
        }
        if system:
            payload["messages"].append({"role": "system", "content": system})
        payload["messages"].append({"role": "user", "content": prompt})
        req = urllib.request.Request(
            f"{self.local_llm_base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with langfuse_generation(
            name="ollama.complete_structured",
            model=self.local_llm_model,
            input_data={"prompt": prompt, "system": system or ""},
            metadata={"llm_mode": "LOCAL_OSS_LLM", "node": _CURRENT_LLM_NODE.get()},
        ) as generation:
            with urllib.request.urlopen(req, timeout=3) as resp:
                raw = json.loads(resp.read().decode("utf-8", "replace") or "{}")
            content = raw.get("message", {}).get("content", "")
            if not isinstance(content, str):
                raise LLMFailureError("Local LLM response was not valid JSON content", transient=True)
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                result = schema.model_validate(parsed)
                self.last_mode = "LOCAL_OSS_LLM"
                duration_ms = (time.time() - start) * 1000
                input_tokens = self._approx_tokens(prompt + (system or ""))
                output_tokens = self._approx_tokens(content)
                self._record_usage(
                    duration_ms,
                    input_tokens,
                    output_tokens,
                    llm_mode="LOCAL_OSS_LLM",
                )
                if generation is not None:
                    generation.update(
                        output={"content": self._secure(content)},
                        usage_details={"input": input_tokens, "output": output_tokens},
                    )
                return result
        raise LLMFailureError("Local LLM JSON output did not match the expected schema", transient=True)

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
        calls_by_node: Dict[str, Dict[str, Any]] = {}
        for call in self._usage_calls:
            node = call["node"]
            aggregate = calls_by_node.setdefault(node, {
                "calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "duration_ms": 0.0,
                "estimated_cost_usd": 0.0,
                "llm_modes": [],
            })
            aggregate["calls"] += 1
            aggregate["tokens_in"] += call["tokens_in"]
            aggregate["tokens_out"] += call["tokens_out"]
            aggregate["duration_ms"] += call["duration_ms"]
            aggregate["estimated_cost_usd"] += call["estimated_cost_usd"]
            if call["llm_mode"] not in aggregate["llm_modes"]:
                aggregate["llm_modes"].append(call["llm_mode"])
        for aggregate in calls_by_node.values():
            aggregate["duration_ms"] = round(aggregate["duration_ms"], 2)
            aggregate["estimated_cost_usd"] = round(
                aggregate["estimated_cost_usd"], 6
            )
        return {
            "model": self.model_name,
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "duration_ms": round(self.total_duration_ms, 2),
            "estimated_cost_usd": self.total_cost_usd,
            "usage_is_estimated": True,
            "llm_available": self.is_available(),
            "secrets_redacted": self.total_redactions,
            "redaction_categories": dict(self.redaction_categories),
            "calls_by_node": calls_by_node,
            "calls_detail": list(self._usage_calls),
        }

    def is_available(self) -> bool:
        """True when any real LLM tier is available."""
        return self._probe_gemini() or self._probe_local_llm()

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return round(
            (input_tokens / 1000.0) * self._cost_per_1k_input
            + (output_tokens / 1000.0) * self._cost_per_1k_output,
            6,
        )

    def _record_usage(
        self,
        duration_ms: float,
        input_tokens: int,
        output_tokens: int,
        llm_mode: Optional[str] = None,
    ):
        mode = llm_mode or self.last_mode
        estimated_cost = self._estimate_cost(input_tokens, output_tokens) if mode == "LIVE_GEMINI" else 0.0
        self.total_duration_ms += duration_ms
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd = round(self.total_cost_usd + estimated_cost, 6)
        self.call_count += 1
        self._usage_calls.append({
            "node": _CURRENT_LLM_NODE.get(),
            "model": self.local_llm_model if mode == "LOCAL_OSS_LLM" else self.model_name,
            "llm_mode": mode,
            "tokens_in": input_tokens,
            "tokens_out": output_tokens,
            "usage_is_estimated": True,
            "duration_ms": round(duration_ms, 2),
            "estimated_cost_usd": estimated_cost,
        })

    @staticmethod
    def _approx_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def _langfuse_config(self) -> Dict[str, Any]:
        mode = self.last_mode
        model = self.model_name if mode == "LIVE_GEMINI" else self.local_llm_model
        return {
            "tags": ["repointel", mode],
            "metadata": {
                "llm_mode": mode,
                "node": _CURRENT_LLM_NODE.get(),
                "model": model,
            },
        }

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
            self.last_mode = "DETERMINISTIC_FALLBACK"
            self._record_usage(
                0,
                self._approx_tokens(prompt + (system or "")),
                self._approx_tokens(self.DEGRADED_MODE_MARKER),
                llm_mode="DETERMINISTIC_FALLBACK",
            )
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
            self.last_mode = "LIVE_GEMINI"
            out = str(llm.invoke(messages, config=self._langfuse_config()).content)
            self._record_usage(
                (time.time() - start) * 1000,
                self._approx_tokens(prompt),
                self._approx_tokens(out),
                llm_mode="LIVE_GEMINI",
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

    def _generate_degraded_plan(self, prompt: str, schema: Type[T]) -> T:
        """Build a deterministic plan tailored to the requested artifact."""
        from packages.agentic.state.schema import PlanTask

        artifact_type = "README"
        if "Target Artifact:" in prompt:
            artifact_type = prompt.split("Target Artifact:", 1)[1].splitlines()[0].strip()
        normalized_type = artifact_type.upper().replace(".MD", "")

        archetype = "repository"
        if "Archetype:" in prompt:
            archetype = prompt.split("Archetype:", 1)[1].splitlines()[0].strip() or archetype

        plan_templates = {
            "README": [
                ("Overview", "project purpose and entry points", "code"),
                ("Installation", "declared setup and dependency instructions", "manifest"),
                ("Usage", "public usage examples and command entry points", "code"),
                ("Configuration", "configuration files and environment settings", "code"),
                ("Testing", "test layout and test commands", "tests"),
            ],
            "ARCHITECTURE": [
                ("System Overview", "major system responsibilities", "code"),
                ("Components", "module boundaries and key components", "code"),
                ("Data Flow", "calls and data flow between components", "symbol_graph"),
                ("Dependencies", "declared runtime and development dependencies", "manifest"),
                ("Runtime Flow", "application startup and execution flow", "code"),
                ("Configuration", "runtime configuration sources", "code"),
            ],
            "API_DOCUMENTATION": [
                ("Endpoints", "registered routes and handlers", "code"),
                ("Request Models", "request schemas and validation", "code"),
                ("Response Models", "response schemas and return values", "code"),
                ("Authentication", "authentication and authorization checks", "code"),
                ("Errors", "error responses and exception handling", "code"),
                ("Examples", "API examples present in tests or documentation", "tests"),
            ],
            "DEVELOPER_GUIDE": [
                ("Setup", "declared developer setup and dependencies", "manifest"),
                ("Development Workflow", "development commands and entry points", "code"),
                ("Code Standards", "configured linters and formatting conventions", "manifest"),
                ("Debugging", "logging and debugging facilities", "code"),
                ("Testing", "test layout and test commands", "tests"),
            ],
            "CONFIGURATION_GUIDE": [
                ("Environment Variables", "environment variable definitions and defaults", "code"),
                ("Config Files", "configuration files and loading behavior", "code"),
                ("Database Settings", "database configuration and migrations", "code"),
                ("Third-Party Credentials", "credential configuration names, never secret values", "code"),
            ],
            "DEPENDENCY_REPORT": [
                ("Runtime Dependencies", "runtime dependency manifests", "manifest"),
                ("Dev Dependencies", "development and test dependencies", "manifest"),
                ("Frameworks", "frameworks declared or imported by the project", "manifest"),
                ("Vulnerabilities and Ecosystems", "dependency ecosystems and available security checks", "manifest"),
            ],
            "CODEBASE_MAP": [
                ("Directory Structure", "top-level directories and their responsibilities", "code"),
                ("Key Modules", "important modules and their responsibilities", "code"),
                ("Symbol Map", "important symbol relationships", "symbol_graph"),
                ("Entrypoints", "application and command entry points", "code"),
            ],
            "ONBOARDING_GUIDE": [
                ("Quickstart", "declared setup and first-run instructions", "manifest"),
                ("Repository Layout", "major directories and modules", "code"),
                ("Key Workflows", "core runtime workflows", "code"),
                ("Best Practices", "documented development and testing practices", "tests"),
            ],
            "CHANGE_SUMMARY": [
                ("Recent Commit Summaries", "recent repository commit history", "git_history"),
                ("Modified Symbols", "symbols changed in available history", "git_history"),
                ("Diff Impact", "impact of recorded changes", "symbol_graph"),
                ("Version Changes", "version and dependency changes", "manifest"),
            ],
        }
        templates = plan_templates.get(normalized_type, [
            ("Overview", "repository purpose and entry points", "code"),
            ("Components", "key repository components", "code"),
            ("Configuration", "configuration files and settings", "code"),
        ])
        if normalized_type == "README":
            archetype_name = archetype.casefold().replace("_", "-")
            if any(term in archetype_name for term in ("cli", "command-line")):
                templates = [
                    ("Overview", "CLI purpose and entry points", "code"),
                    ("Installation", "declared setup and dependency instructions", "manifest"),
                    ("Commands", "registered commands and their arguments", "code"),
                    ("Usage", "actual CLI invocation examples", "code"),
                    ("Configuration", "configuration files and environment settings", "code"),
                    ("Testing", "test layout and test commands", "tests"),
                ]
            elif "librar" in archetype_name or "sdk" in archetype_name:
                templates = [
                    ("Overview", "library purpose and supported capabilities", "code"),
                    ("Installation", "declared package installation instructions", "manifest"),
                    ("API Reference", "public symbols and interfaces", "symbol_graph"),
                    ("Examples", "usage examples present in repository files", "tests"),
                    ("Development", "development setup and contribution workflow", "manifest"),
                    ("Testing", "test layout and test commands", "tests"),
                ]
            elif "monorepo" in archetype_name or "multi-package" in archetype_name:
                templates = [
                    ("Overview", "repository purpose and top-level entry points", "code"),
                    ("Repository Structure", "packages and their responsibilities", "code"),
                    ("Packages", "package manifests and dependency relationships", "manifest"),
                    ("Development Setup", "workspace setup and development commands", "manifest"),
                    ("Testing", "test layout and test commands", "tests"),
                ]
            elif any(term in archetype_name for term in ("service", "web", "api")):
                templates = [
                    ("Overview", "service purpose and entry points", "code"),
                    ("Installation", "declared setup and dependency instructions", "manifest"),
                    ("API", "registered endpoints and request handlers", "code"),
                    ("Usage", "service usage examples and client entry points", "code"),
                    ("Configuration", "configuration files and environment settings", "code"),
                    ("Testing", "test layout and test commands", "tests"),
                ]
        tasks = [
            PlanTask(
                task_id=f"degraded-{index}",
                objective=f"Find evidence for {section}: {objective} in this {archetype}.",
                retrieval_query=objective,
                evidence_type=evidence_type,
                target_section=section,
                priority=index,
            )
            for index, (section, objective, evidence_type) in enumerate(templates, start=1)
        ]

        feedback_marker = "Review Feedback from previous runs (if any):"
        if feedback_marker in prompt:
            feedback_text = prompt.split(feedback_marker, 1)[1].split(
                "Please produce", 1
            )[0].strip()
            try:
                feedback = json.loads(feedback_text)
            except (TypeError, ValueError):
                feedback = None
            if isinstance(feedback, dict):
                planned_sections = {task.target_section.casefold() for task in tasks}
                for issue in feedback.get("issues", []):
                    if not isinstance(issue, dict):
                        continue
                    section = str(issue.get("target_section") or "").strip()
                    is_coverage_gap = str(issue.get("issue_type", "")).upper() == "COVERAGE_GAP"
                    is_high_severity = str(issue.get("severity", "")).upper() == "HIGH"
                    if section and (is_coverage_gap or is_high_severity) and section.casefold() not in planned_sections:
                        description = str(issue.get("description") or "review feedback")
                        tasks.append(PlanTask(
                            task_id=f"degraded-{len(tasks) + 1}",
                            objective=f"Address review feedback for {section}: {description}",
                            retrieval_query=f"{section} {description}",
                            evidence_type="code",
                            target_section=section,
                            priority=len(tasks) + 1,
                        ))
                        planned_sections.add(section.casefold())

        return schema(
            plan_id="degraded-plan",
            repo_id="",
            artifact_type=artifact_type,
            document_outline=[task.target_section for task in tasks],
            tasks=tasks,
        )


    def _generate_tool_call_decision(self, prompt: str):
        """Deterministic fallback for ToolCallDecision.

        Select the deterministic tool best suited to the declared evidence type,
        and stop when the citation or round budget is exhausted.
        """
        from packages.agentic.core.tool_loop import ToolCallDecision

        context_match = re.search(
            r"BEGIN UNTRUSTED REPOSITORY DATA [a-f0-9]{32}\n(.*?)\n"
            r"END UNTRUSTED REPOSITORY DATA [a-f0-9]{32}",
            prompt,
            re.DOTALL,
        )
        try:
            task_context = json.loads(context_match.group(1)) if context_match else {}
        except (TypeError, ValueError):
            task_context = {}
        target = str(task_context.get("target_section") or "")
        query = str(task_context.get("retrieval_query") or "")
        evidence_type = str(task_context.get("evidence_type") or "").lower().replace("-", "_")

        def field_value(name: str) -> str:
            match = re.search(rf"(?m)^\s*{re.escape(name)}:\s*([^\r\n]+)", prompt)
            return match.group(1).strip() if match else ""

        try:
            gathered = int(field_value("Citations Gathered So Far") or "0")
        except ValueError:
            gathered = 0
        try:
            current_round, max_rounds = (
                int(part.strip()) for part in field_value("Current Round").split("/")
            )
        except (ValueError, TypeError):
            current_round, max_rounds = 1, 4

        if gathered >= 3 or current_round >= max_rounds:
            return ToolCallDecision(
                tool_name="finish_task",
                query_or_param="",
                reasoning="Deterministic: citation or retrieval-round budget reached",
            )
        if evidence_type == "manifest" or any(
            keyword in f"{target} {query}".lower()
            for keyword in ("dependency", "dependencies", "installation", "install")
        ):
            return ToolCallDecision(
                tool_name="get_dependency_manifest",
                query_or_param="",
                reasoning="Deterministic: task requests manifest evidence",
            )
        if evidence_type == "file_metadata":
            candidate = query.strip(" `'")
            if "/" in candidate or "\\" in candidate or "." in candidate:
                return ToolCallDecision(
                    tool_name="get_file_metadata",
                    query_or_param=candidate,
                    reasoning="Deterministic: task requests metadata for a file path",
                )
        if evidence_type == "symbol_graph":
            tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_.]*\b", query)
            candidate = next((
                token for token in tokens
                if ("." in token or any(char.isupper() for char in token[1:]))
            ), "")
            if candidate:
                return ToolCallDecision(
                    tool_name="symbol_graph_neighbors",
                    query_or_param=candidate,
                    reasoning="Deterministic: task requests symbol relationship evidence",
                )
        return ToolCallDecision(
            tool_name="hybrid_retrieve",
            query_or_param=query or target or "repository code",
            reasoning="Deterministic: retrieve code evidence for the current task",
        )

    def complete_structured(self, prompt: str, schema: Type[T], system: Optional[str] = None, max_retries: int = 2) -> T:
        if self._probe_gemini():
            try:
                start = time.time()
                from langchain_google_genai import ChatGoogleGenerativeAI
                prompt_text = self._secure(prompt)
                system_text = self._secure(system) if system else None
                llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    thinking_level="medium",
                    temperature=0.2,
                    max_retries=1,
                )
                structured_llm = llm.with_structured_output(schema)
                messages = []
                if system_text:
                    messages.append(SystemMessage(content=system_text))
                messages.append(HumanMessage(content=prompt_text))
                self.last_mode = "LIVE_GEMINI"

                for attempt in range(max_retries + 1):
                    try:
                        res = None
                        for chunk in structured_llm.stream(
                            messages,
                            config=self._langfuse_config(),
                        ):
                            res = chunk
                        self.last_mode = "LIVE_GEMINI"
                        self._record_usage(
                            (time.time() - start) * 1000,
                            self._approx_tokens(prompt_text),
                            self._approx_tokens(str(res)),
                            llm_mode="LIVE_GEMINI",
                        )
                        if isinstance(res, schema):
                            return res
                        if isinstance(res, dict):
                            return schema.model_validate(res)
                    except Exception as e:
                        self.last_error = str(e)
                        msg_l = str(e).lower()
                        transient_markers = ("429", "503", "rate", "quota", "timeout", "connection", "internal", "unavailable")
                        if any(m in msg_l for m in transient_markers):
                            if attempt < max_retries:
                                time.sleep(6 * (attempt + 1))
                                continue
                            break
                        messages.append(HumanMessage(content=f"[Validation error: {str(e)}]"))
                raise LLMFailureError(f"Gemini structured completion failed after retries", transient=True)
            except Exception as exc:
                if isinstance(exc, LLMFailureError):
                    if self._probe_local_llm():
                        self.last_mode = "LOCAL_OSS_LLM"
                        return self._call_local_structured_model(prompt, schema, system)
                    return self._complete_deterministic_structured(prompt, schema)
                if self._probe_local_llm():
                    self.last_mode = "LOCAL_OSS_LLM"
                    return self._call_local_structured_model(prompt, schema, system)
                return self._complete_deterministic_structured(prompt, schema)

        if self._probe_local_llm():
            self.last_mode = "LOCAL_OSS_LLM"
            return self._call_local_structured_model(prompt, schema, system)

        return self._complete_deterministic_structured(prompt, schema)

    def _complete_deterministic_structured(self, prompt: str, schema: Type[T]) -> T:
        start = time.time()
        result = self._deterministic_structured_fallback(prompt, schema)
        self.last_mode = "DETERMINISTIC_FALLBACK"
        self._record_usage(
            (time.time() - start) * 1000,
            self._approx_tokens(prompt),
            self._approx_tokens(str(result)),
            llm_mode="DETERMINISTIC_FALLBACK",
        )
        return result

    def _deterministic_structured_fallback(self, prompt: str, schema: Type[T]) -> T:
        from packages.agentic.state.schema import ReviewVerdict, EvidenceCitation
        if schema.__name__ == "Plan":
            return self._generate_degraded_plan(prompt, schema)
        if schema.__name__ == "ToolCallDecision":
            return self._generate_tool_call_decision(prompt)
        if schema.__name__ == "ArtifactSection":
            return self._generate_artifact_section(prompt)
        if schema.__name__ == "ReviewReport":
            return schema(verdict=ReviewVerdict.APPROVE, issues=[], groundedness_score=1.0, coverage_score=1.0)
        if schema.__name__ == "EvidenceCitation":
            return schema(file_path="", content="", line_start=0, line_end=0)
        return schema.model_construct()
