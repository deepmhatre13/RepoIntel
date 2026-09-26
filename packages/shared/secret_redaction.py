"""Secret detection and redaction for untrusted repository content.

RepoIntel analyzes third-party repositories that may contain live
credentials (checked-in ``.env`` files, hardcoded API keys, private keys).
Before *any* repository-derived text is sent to an external service such as
Gemini, obvious secrets must be replaced with typed placeholders.

Design rules
------------
* Detection is heuristic and intentionally conservative on **what it
  redacts** (better to over-redact than leak).
* The raw secret value never appears in logs, telemetry, errors, or
  redaction reports -- only the category name and count.
* Placeholders keep evidence auditable::

      [REDACTED:GITHUB_TOKEN]

* This module performs pure static text processing.  It never executes
  anything from the analyzed repository.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple


# ---------------------------------------------------------------------------
# Compiled detection patterns: (category, compiled regex)
# ---------------------------------------------------------------------------

def _compile_patterns() -> List[Tuple[str, str]]:
    return [
        # --- Cloud provider keys -------------------------------------------
        ("AWS_ACCESS_KEY_ID", r"\b(AKIA[0-9A-Z]{16})\b"),
        ("GOOGLE_API_KEY", r"\b(AIza[0-9A-Za-z_\-]{35})\b"),
        # --- Service-specific tokens ---------------------------------------
        ("GITHUB_TOKEN", r"\b(gh[pousr]_[A-Za-z0-9]{30,})\b"),
        ("SLACK_TOKEN", r"\b(xox[baprs]-[A-Za-z0-9\-]{10,})\b"),
        ("STRIPE_SECRET_KEY", r"\b((?:sk|rk)_live_[A-Za-z0-9]{16,})\b"),
        ("OPENAI_API_KEY", r"\b(sk-[A-Za-z0-9_\-]{20,})\b"),
        ("ANTHROPIC_API_KEY", r"\b(sk-ant-[A-Za-z0-9_\-]{20,})\b"),
        # --- Tokens / sessions ---------------------------------------------
        ("BEARER_TOKEN", r"(?i)\b(bearer\s+[A-Za-z0-9_\-._~+/]{20,}=*)"),
        ("JWT", r"\b(eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})\b"),
        # --- Private key material ------------------------------------------
        (
            "PRIVATE_KEY_BLOCK",
            r"(?s)(-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----)",
        ),
        # --- Key/value assignments -----------------------------------------
        #   SECRET_KEY=..., api_token: "...", DB_PASSWORD = '...'
        (
            "ASSIGNED_CREDENTIAL",
            r"(?im)^[\s\-*]*[\"']?[A-Za-z0-9_]*(?:api_key|apikey|secret|token|"
            r"password|passwd|pwd|credentials?|access_key|private_key|auth_key)"
            r"[A-Za-z0-9_]*[\"']?\s*[:=]\s*[\"']?([^\s\"'#{;,]{8,})",
        ),
        # --- URLs with inline credentials -----------------------------------
        ("URL_CREDENTIALS", r"\b([a-zA-Z][a-zA-Z0-9+.\-]*://[^:/\s@]+:)[^@\s/]+@"),
    ]


_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    (name, re.compile(rx)) for name, rx in _compile_patterns()
]

# Values that look like placeholders already (e.g. os.environ references,
# template variables) must not be treated as live secrets.
_BENIGN_VALUE = re.compile(
    r"^(?:\$\{?[A-Za-z0-9_]+\}?|<[A-Za-z0-9_\-]+>|\{\{[^}]+\}\}|None|null|true|false|"
    r"os\.environ.*|getenv|getenv\(.*|your[-_].*|xxx+|\*+)$",
    re.IGNORECASE,
)


@dataclass
class RedactionResult:
    """Outcome of scanning/redacting one piece of text."""

    text: str
    total_redactions: int = 0
    categories: Dict[str, int] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return self.total_redactions == 0


def _is_benign(value: str) -> bool:
    v = value.strip().strip("\"'")
    if not v or len(v) < 6:
        return True
    return bool(_BENIGN_VALUE.match(v))


def scan_for_secrets(text: str) -> Dict[str, int]:
    """Return ``{category: occurrences}`` without modifying ``text``.

    Used by audits; counts only -- never includes matched values.
    """
    found: Dict[str, int] = {}
    for category, pattern in _PATTERNS:
        n = 0
        for match in pattern.finditer(text):
            groups = [g for g in match.groups() if g is not None]
            candidate = groups[-1] if groups else match.group(0)
            if category == "URL_CREDENTIALS" or not _is_benign(candidate):
                n += 1
        if n:
            found[category] = found.get(category, 0) + n
    return found


def redact_secrets(text: str) -> RedactionResult:
    """Replace detected secrets in ``text`` with typed placeholders.

    Returns a :class:`RedactionResult`; the original secret values are
    discarded and never surfaced.
    """
    if not text:
        return RedactionResult(text=text)

    result_text = text
    categories: Dict[str, int] = {}

    for category, pattern in _PATTERNS:

        def _replace(match: "re.Match[str]", _cat: str = category) -> str:
            groups = [g for g in match.groups() if g is not None]
            if _cat == "URL_CREDENTIALS":
                return match.group(1) + "[REDACTED:URL_CREDENTIALS]@"
            if len(groups) >= 2:
                # Keep the key name (e.g. ``DB_PASSWORD=``), redact value only.
                kept = match.group(0)[: match.start(len(groups))]
                return f"{kept}[REDACTED:{_cat}]"
            if _is_benign(groups[0] if groups else match.group(0)):
                return match.group(0)
            return f"[REDACTED:{_cat}]"

        new_text, _n_subs = pattern.subn(_replace, result_text)
        before = result_text.count("[REDACTED:")
        after = new_text.count("[REDACTED:")
        actual = max(0, after - before)
        if actual:
            categories[category] = categories.get(category, 0) + actual
            result_text = new_text

    return RedactionResult(
        text=result_text,
        total_redactions=sum(categories.values()),
        categories=categories,
    )
