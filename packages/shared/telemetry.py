import json
import time
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("RepoIntel.Telemetry")

class TelemetryTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.node_telemetry = []
        self.stage_durations: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self.events: List[Dict[str, Any]] = []

    def log_node_execution(self, node_name: str, duration_ms: float, input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0, status: str = "SUCCESS", error: Optional[str] = None):
        entry = {
            "node_name": node_name,
            "duration_ms": duration_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": cost_usd,
            "status": status,
            "error": error
        }
        self.node_telemetry.append(entry)
        logger.info(f"NODE_TELEMETRY: {json.dumps(entry)}")

    # --- Phase 42.7 / Observability extensions ---

    def record_stage_duration(self, stage: str, duration_seconds: float):
        self.stage_durations[stage] = round(self.stage_durations.get(stage, 0.0) + duration_seconds, 4)

    def increment(self, counter: str, amount: int = 1):
        self.counters[counter] = self.counters.get(counter, 0) + amount

    def record_event(self, kind: str, **details):
        """Structured event (cache_hit, cache_invalidation, retry, failure, recovery...)."""
        entry = {"kind": kind, "ts": time.time(), **details}
        self.events.append(entry)
        logger.info(f"REPOINTEL_EVENT: {json.dumps(entry)}")

    def get_summary(self) -> Dict[str, Any]:
        total_duration = sum(t["duration_ms"] for t in self.node_telemetry)
        total_in = sum(t["input_tokens"] for t in self.node_telemetry)
        total_out = sum(t["output_tokens"] for t in self.node_telemetry)
        total_cost = sum(t["estimated_cost_usd"] for t in self.node_telemetry)
        return {
            "total_nodes_executed": len(self.node_telemetry),
            "total_duration_ms": total_duration,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_estimated_cost_usd": round(total_cost, 6),
            "stage_durations_seconds": dict(self.stage_durations),
            "counters": dict(self.counters),
            "events": list(self.events),
            "nodes": self.node_telemetry
        }

    def snapshot_counters(self) -> Dict[str, Any]:
        return {
            "stage_durations_seconds": dict(self.stage_durations),
            "counters": dict(self.counters),
            "event_count": len(self.events),
        }

telemetry = TelemetryTracker()
