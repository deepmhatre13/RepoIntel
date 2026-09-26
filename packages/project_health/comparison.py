"""Project Health comparison (Stage 4B).

Deterministic, evidence-backed comparison of two repositories' persisted health
snapshots. Uses the SAME ``HealthAnalysisEngine.METRIC_DEFS`` normalization as
scoring, so a value compared across repositories is the value that was scored.
No LLM involvement: the ranking and interpretation rules are fixed functions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from packages.project_health.analysis import HealthAnalysisEngine
from packages.project_health.models import HealthEvidence  # noqa: F401 (typed evidence)


# Metrics surfaced in the comparison grid: key -> (dimension, label).
# Chosen to cover maintenance, activity, community, releases, dependency and
# code health, plus security via the source snapshot details.
_COMPARE_METRICS = [
    ("rolling_30d", "activity", "Commits (30d window)"),
    ("active_contributors_30d", "community", "Active contributors (30d)"),
    ("hhi", "community", "Contributor concentration (HHI)"),
    ("merge_rate", "maintenance", "PR merge rate"),
    ("median_merge_days", "maintenance", "PR response (median days)"),
    ("median_interval_days", "release", "Release interval (median days)"),
    ("total_dependencies", "dependency", "Total dependencies"),
    ("findings_count", "security", "Security findings"),
    ("files_parsed", "code", "Files understood by source analysis"),
]

engine = HealthAnalysisEngine()


def _evidence_index(analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Flatten a snapshot's dimension evidence into {metric: evidence-dict}.

    Evidence is kept as the persisted dict (no model reconstruction), so
    nothing is fabricated or dropped when a field is absent.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for d in analysis.get("dimensions") or []:
        for e in d.get("evidence") or []:
            m = e.get("metric")
            if m and m not in out:
                out[m] = e
    return out


def _interpret(metric: str, a: Optional[float], b: Optional[float],
               higher: bool) -> str:
    """Fixed, evidence-grounded reading of a difference. No invented cause."""
    if a is None or b is None:
        return "not comparable - a value is missing for one repository"
    diff = a - b
    if abs(diff) < 1e-9:
        return "equal"
    good_a = diff > 0 if higher else diff < 0
    return ("A is favorable" if good_a else "B is favorable") + \
        f" ({'higher' if diff > 0 else 'lower'} by {round(abs(diff), 3)})"


def compare_project_health(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two persisted health snapshots deterministically.

    Both inputs are snapshot dicts (as persisted by ``save_analysis``).
    Missing metrics are reported explicitly rather than treated as zero.
    """
    ev_a = _evidence_index(a)
    ev_b = _evidence_index(b)
    rows: List[Dict[str, Any]] = []
    for metric, dim, label in _COMPARE_METRICS:
        spec = engine.METRIC_DEFS.get(metric)
        if not spec:
            continue
        higher = spec[1]
        ea, eb = ev_a.get(metric), ev_b.get(metric)
        va, vb = ((ea or {}).get("value"), (eb or {}).get("value"))
        rows.append({
            "metric": metric,
            "label": label,
            "dimension": dim,
            "repository_a": va,
            "repository_b": vb,
            "difference": (round(va - vb, 4)
                           if va is not None and vb is not None else None),
            "interpretation": _interpret(metric, va, vb, higher),
            "evidence": {
                "a": {"source": (ea or {}).get("source"),
                      "window": (ea or {}).get("window"),
                      "availability": (ea or {}).get("availability", "NOT_AVAILABLE")},
                "b": {"source": (eb or {}).get("source"),
                      "window": (eb or {}).get("window"),
                      "availability": (eb or {}).get("availability", "NOT_AVAILABLE")},
            },
        })

    def _dim_map(s: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {d.get("name"): d for d in s.get("dimensions") or []}

    dims_a, dims_b = _dim_map(a), _dim_map(b)
    dim_rows = []
    for name in ["maintenance", "community", "release", "activity",
                 "dependency", "security", "code"]:
        da, db = dims_a.get(name), dims_b.get(name)
        sa = da.get("score") if da else None
        sb = db.get("score") if db else None
        diff = (round(sa - sb, 1) if sa is not None and sb is not None else None)
        dim_rows.append({
            "dimension": name,
            "score_a": sa, "score_b": sb, "difference": diff,
            "state_a": da.get("state") if da else "NOT_AVAILABLE",
            "state_b": db.get("state") if db else "NOT_AVAILABLE",
            "interpretation": _interpret(name, sa, sb, True),
        })

    traj_a, traj_b = a.get("trajectory"), b.get("trajectory")
    return {
        "repository_a": a.get("full_name") or a.get("url"),
        "repository_b": b.get("full_name") or b.get("url"),
        "trajectory": {
            "a": traj_a, "b": traj_b,
            "interpretation": (
                "both repositories share the same trajectory" if traj_a == traj_b
                else f"A: {traj_a} vs B: {traj_b} (statistical, not causal)"),
        },
        "overall_score": {
            "a": a.get("overall_score"), "b": b.get("overall_score"),
            "state_a": a.get("score_state"), "state_b": b.get("score_state"),
        },
        "dimensions": dim_rows,
        "metrics": rows,
    }
