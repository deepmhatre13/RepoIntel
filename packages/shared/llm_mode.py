from typing import Literal

LLMMode = Literal[
    "LIVE_GEMINI",
    "LOCAL_OSS_LLM",
    "DETERMINISTIC_FALLBACK",
    "DEGRADED_FALLBACK",
    "LLM_UNAVAILABLE",
]