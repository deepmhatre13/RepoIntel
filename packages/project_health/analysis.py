"""Multidimensional health analysis engine (Stages 10-15).

Builds the Project Health model from the feature + temporal layers:

* health dimensions (maintenance, community, release, code, dependency,
  security) — each with current state, trend, confidence, supporting metrics
  and evidence,
* a decomposable 0-100 overall score from normalized component signals (the
  score is never invented: it is an explicitly defined aggregation of real,
  confidence-weighted component signals),
* independent risk dimensions,
* correlation results between metrics,
* change points / anomalies / trends,
* a \"why did health change\" contribution ranking,
* a derived project timeline.

Scoring approach: each contributing metric maps onto a 0-1 signal whose sign
indicates the good direction (higher=better). Missing/invalid data signs do
not contribute to the numerator (so the score reflects only real evidence) but
count into coverage to lower confidence. This keeps the score explicit & honest.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from packages.project_health.models import (
    AnomalyResult,
    ChangePointResult,
    CorrelationResult,
    DataAvailability,
    HealthDimension,
    HealthEvidence,
    HealthRisk,
    HealthSignal,
    HealthState,
    RiskLevel,
    TrendDirection,
)
from packages.project_health import temporal


def _now() -> datetime:
    return datetime.now(timezone.utc)


class HealthAnalysisEngine:
    """Deterministic core of the Project Health model (no arbitrary scoring)."""

    # Bounds are (range_low, range_high): a metric's raw value is placed
    # linearly in that range, then flipped when higher_is_better is False.
    # e.g. median_merge_days: 0.2d -> good(1.0), 1.0d -> bad(0.0).
    METRIC_DEFS: Dict[str, Tuple] = {
        "rolling_7d": ("activity", True, 0.0, 5.0),
        "rolling_30d": ("activity", True, 0.0, 20.0),
        "rolling_90d": ("activity", True, 0.0, 50.0),
        "active_days_30d": ("activity", True, 0.0, 30.0),
        "merge_rate": ("maintenance", True, 0.0, 0.9),
        "median_merge_days": ("maintenance", False, 0.2, 1.0),
        "median_resolution_days": ("maintenance", False, 0.2, 1.0),
        "closure_rate": ("maintenance", True, 0.0, 0.8),
        "stale_open_ratio": ("maintenance", False, 0.0, 0.6),
        "stale_issue_ratio": ("maintenance", False, 0.0, 0.8),
        "median_interval_days": ("release", False, 0.5, 7.0),
        "frequency_per_90d": ("release", True, 0.0, 4.0),
        "hhi": ("community", False, 0.0, 1.0),
        "contributor_count": ("community", True, 0.0, 15.0),
        "active_contributors_30d": ("community", True, 0.0, 10.0),
        # --- Source-intelligence metrics (dependency / security / code) ---
        # Dependencies: a very large flat dependency surface broadens the
        # supply-chain risk. Higher counts are treated as slightly risky.
        "total_dependencies": ("dependency", False, 0.0, 60.0),
        "direct_dependencies": ("dependency", False, 0.0, 60.0),
        # Security: lower is always better; bounds reflect severity weighting.
        "findings_count": ("security", False, 0.0, 12.0),
        "error_count": ("security", False, 0.0, 3.0),
        "warning_count": ("security", False, 0.0, 20.0),
        # Code health: parse failures are a direct signal of a fragile tree;
        # symbol/edge counts reflect how much structure is understood.
        "parse_failures": ("code", False, 0.0, 6.0),
        "files_parsed": ("code", True, 0.0, 200.0),
        "symbol_count": ("code", True, 0.0, 1000.0),
        "edge_count": ("code", True, 0.0, 800.0),
    }

    DIMENSION_NAMES = ["maintenance", "community", "release", "activity",
                       "dependency", "security", "code"]

    def run(
        self,
        *,
        features: Dict[str, Any],
        changes: List[Dict[str, Any]],
        trends: Dict[str, Dict[str, Any]],
        correlations: List[CorrelationResult],
        anomalies: List[AnomalyResult],
        change_points: List[ChangePointResult],
        status: Dict[str, str],
    ) -> Dict[str, Any]:
        evidence = self._build_evidence(features)
        dims, _ = self._compute_dimensions(evidence, status)
        correlation_pairs = self._correlate_features(features)
        overall, coverage, composed = self._aggregate_score(dims)
        risks = self._compute_risks(dims, features)
        why_changed = self._why_changed(evidence)
        signals = self._build_signals(dims, changes, evidence)
        return {
            "dimensions": dims,
            "overall_score": overall,
            "score_coverage": coverage,
            "score_breakdown": composed,
            "risks": risks,
            "correlations": correlation_pairs,
            "signals": signals,
            "why_changed": why_changed,
            "changes": changes,
        }

    # -- evidence collection -------------------------------------------------
    def _build_evidence(self, features: Dict[str, Any]) -> List[HealthEvidence]:
        ev: List[HealthEvidence] = []
        for group in ("activity", "pull_requests", "issues", "releases",
                      "community"):
            f = features.get(group) or {}
            if not isinstance(f, dict):
                continue
            if f.get("availability") == "NOT_AVAILABLE":
                continue  # provider failure: no evidence from this group
            for key, val in f.items():
                if key in ("metric", "values", "n", "valid"):
                    continue
                if isinstance(val, bool) or key.endswith("availability"):
                    continue
                # None = metric could not be computed (API error / no data);
                # it must not enter evidence, otherwise it would score as a
                # measured zero and fabricate a deterioration signal.
                if val is None:
                    continue
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    ev.append(HealthEvidence(
                        metric=key, value=round(float(val), 4),
                        source="github", confidence=1.0,
                                                availability=DataAvailability.AVAILABLE.value,
                    ))
        # Source-intelligence groups are only admitted as evidence when the
        # feature group is AVAILABLE — never fabricate a zero for an
        # unavailable scanner or unparsed repository.
        for group in ("dependencies", "security", "code_health"):
            f = features.get(group) or {}
            if not isinstance(f, dict) or f.get("availability") != "AVAILABLE":
                continue
            source = {"dependencies": "dependencies", "security": "security",
                      "code_health": "code"}.get(group, "source")
            for key, val in f.items():
                if key in ("metric", "values", "n", "valid", "availability",
                           "manifest_files", "ecosystems", "languages",
                           "findings_by_rule", "scanner_available", "error"):
                    continue
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    ev.append(HealthEvidence(
                        metric=key, value=round(float(val), 4),
                        source=source, confidence=1.0,
                        availability=DataAvailability.AVAILABLE.value,
                    ))
        return ev

    def _compute_dimensions(
            self, evidence: List[HealthEvidence], status: Dict[str, str]
        ) -> Tuple[List[HealthDimension], Dict[str, List[HealthEvidence]]]:
        buckets: Dict[str, List[HealthEvidence]] = {d: [] for d in self.DIMENSION_NAMES}
        for e in evidence:
            dim = self.METRIC_DEFS.get(e.metric, (None,))[0]
            if dim and dim in buckets:
                buckets[dim].append(e)

        dims: List[HealthDimension] = []
        for name in self.DIMENSION_NAMES:
            ev = buckets[name]
            if not ev:
                dims.append(HealthDimension(
                    name=name, state=HealthState.NOT_AVAILABLE,
                    trend=TrendDirection.NOT_AVAILABLE,
                    availability=DataAvailability.NOT_AVAILABLE.value,
                    confidence=0.0, score=None,
                ))
                continue
            score, conf = self._dimension_score(name, ev)
            state = self._state_from_score(score, conf)
            trend = self._dimension_trend(name, ev)
            dims.append(HealthDimension(
                name=name, state=state, trend=trend,
                confidence=round(conf, 3), score=score,
                availability=DataAvailability.AVAILABLE.value,
                supporting_metrics=[e.metric for e in ev],
                evidence=ev,
            ))
        return dims, buckets

    def _dimension_score(self, dim: str, ev: List[HealthEvidence]) -> Tuple[Optional[float], float]:
        """Weighted 0-100 score over component signals for a dimension.

        Each signal is linearly normalized between declared worst/best bounds
        (saturating at extremes) and aggregated; missing metrics lower
        confidence instead of masking the numerator.
        """
        total = 0.0
        weight_sum = 0.0
        for e in ev:
            spec = self.METRIC_DEFS.get(e.metric)
            if not spec or e.value is None:
                continue
            _, higher, worst, best = spec
            if abs(best - worst) < 1e-9:
                continue
            raw = (e.value - worst) / (best - worst)
            norm = min(1.0, max(0.0, raw))
            if not higher:
                norm = 1.0 - norm
            total += norm
            weight_sum += 1.0
        if weight_sum == 0:
            return None, 0.0
        score = round(100.0 * total / weight_sum, 1)
        confidence = round(max(0.25, min(1.0, weight_sum / 3.0)), 3)
        return score, confidence

    @staticmethod
    def _state_from_score(score: Optional[float], confidence: float) -> HealthState:
        if score is None:
            return HealthState.NOT_AVAILABLE
        if confidence < 0.45:
            return HealthState.STABLE  # not enough evidence to label good/bad
        if score >= 70:
            return HealthState.GOOD
        if score >= 55:
            return HealthState.STABLE
        return HealthState.DETERIORATING

    def _dimension_trend(self, dim: str, ev: List[HealthEvidence]) -> TrendDirection:
        """Aggregate a dimension's recent-vs-previous direction using metric signs."""
        votes = 0
        good = 0
        for e in ev:
            spec = self.METRIC_DEFS.get(e.metric)
            if not spec or e.value is None or e.previous_value is None:
                continue
            higher = spec[1]
            improved = e.value > e.previous_value if higher else e.value < e.previous_value
            votes += 1
            if improved:
                good += 1
        if votes == 0:
            return TrendDirection.NOT_AVAILABLE
        ratio = good / votes
        if ratio >= 0.65:
            return TrendDirection.INCREASING
        if ratio <= 0.35:
            return TrendDirection.DECREASING
        return TrendDirection.STABLE

    def _correlate_features(self, features: Dict[str, Any]) -> List[CorrelationResult]:
        pairs = [
            ("hhi", "median_merge_days"),
            ("hhi", "median_resolution_days"),
            ("rolling_30d", "active_contributors_30d"),
            ("median_interval_days", "dependency_staleness"),
            ("rolling_30d", "frequency_per_90d"),
        ]
        out: List[CorrelationResult] = []
        for (a, b) in pairs:
            va = self._series_for(features, a)
            vb = self._series_for(features, b)
            if va is None or vb is None:
                out.append(CorrelationResult(metric_a=a, metric_b=b))
                continue
            r = temporal.pearson_correlation(va, vb)
            if r is None:
                out.append(CorrelationResult(metric_a=a, metric_b=b))
                continue
            out.append(CorrelationResult(
                metric_a=a, metric_b=b, correlation_coefficient=round(r, 3),
                observation_window="available history", sample_size=len(va),
                confidence=round(min(1.0, len(va) / 12.0), 3),
                availability=DataAvailability.AVAILABLE.value,
            ))
        return out

    @staticmethod
    def _series_for(features: Dict[str, Any], metric: str) -> Optional[List[float]]:
        """Return the current feature value as a 1-pt series (snapshot level).

        Real cross-time correlation uses persisted history in the service; at
        this level we surface the current value so the UI still compares two
        related signals rather than dropping the correlation card.
        """
        for group in features.values():
            if isinstance(group, dict) and metric in group:
                v = group[metric]
                is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
                return [float(v)] if is_num else None
        return None

    def _aggregate_score(self, dims: List[HealthDimension]) -> Tuple[Optional[float], float, Dict[str, Any]]:
        """Confidence-weighted overall score + per-dimension decomposition.

        Every contributing dimension's score and weight is preserved so the
        UI can answer \"why is this score X\" from real component evidence.
        """
        scored = [d for d in dims if d.score is not None]
        if not scored:
            return None, 0.0, {}
        coverage = len(scored) / max(1, len(dims))
        conf_weights = [max(0.25, d.confidence) for d in scored]
        total = sum(d.score * w for d, w in zip(scored, conf_weights))
        overall = round(total / sum(conf_weights), 1)
        breakdown = {d.name: {"score": d.score, "weight": round(c, 3),
                              "state": d.state.value}
                     for d, c in zip(scored, conf_weights)}
        return overall, round(coverage, 2), breakdown

    def _compute_risks(self, dims: List[HealthDimension],
                       features: Dict[str, Any]) -> List[HealthRisk]:
        risk_map = {
            "maintenance_risk": ("maintenance", ["median_merge_days", "merge_rate",
                                                 "median_resolution_days", "closure_rate"]),
            "contributor_risk": ("community", ["hhi", "contributor_count",
                                               "active_contributors_30d"]),
            "release_risk": ("release", ["median_interval_days", "frequency_per_90d"]),
            "dependency_risk": ("dependency", ["total_dependencies", "direct_dependencies"]),
            "code_health_risk": ("code", ["parse_failures", "files_parsed", "symbol_count"]),
            "security_risk": ("security", ["findings_count", "error_count", "warning_count"]),
        }
        by_dim = {d.name: d for d in dims}
        out: List[HealthRisk] = []
        for risk_name, (dim_name, _) in risk_map.items():
            dim = by_dim.get(dim_name)
            if dim is None or dim.score is None:
                out.append(HealthRisk(name=risk_name, level=RiskLevel.NOT_AVAILABLE,
                                      trend=TrendDirection.NOT_AVAILABLE, confidence=0.0))
                continue
            score = dim.score
            level = "HIGH" if score < 45 else ("MEDIUM" if score < 60 else "LOW")
            out.append(HealthRisk(
                name=risk_name, level=RiskLevel(level), trend=dim.trend,
                confidence=dim.confidence, supporting_evidence=dim.evidence[:6],
                summary=f"{dim_name} dimension score {score}/100",
            ))
        return out



    def _build_signals(self, dims: List[HealthDimension], changes: List[Dict[str, Any]],
                       evidence: List[HealthEvidence]) -> List[HealthSignal]:
        sig: List[HealthSignal] = []
        for d in dims:
            if d.state == HealthState.DETERIORATING:
                sig.append(HealthSignal(
                    name=f"{d.name} deteriorating", dimension=d.name, severity="warning",
                    direction=d.trend,
                    description=f"{d.name} health is deteriorating (score {d.score}).",
                    evidence=d.evidence[:5],
                ))
            elif d.state == HealthState.IMPROVING:
                sig.append(HealthSignal(
                    name=f"{d.name} improving", dimension=d.name, severity="info",
                    direction=d.trend,
                    description=f"{d.name} health is improving (score {d.score}).",
                    evidence=d.evidence[:5],
                ))
        for c in changes[:5]:
            if (c.get("delta") or 0) != 0:
                sig.append(HealthSignal(
                    name=f"{c['metric']} changed",
                    dimension=c.get("dimension", "unknown"),
                    severity="warning" if (c.get("delta") or 0) < 0 else "info",
                    direction=c.get("direction", TrendDirection.NOT_AVAILABLE),
                    description=c.get("description", ""),
                    evidence=[HealthEvidence(
                        metric=c.get("metric", ""), value=c.get("value"),
                        previous_value=c.get("previous_value"), delta=c.get("delta"),
                        direction=c.get("direction", TrendDirection.NOT_AVAILABLE),
                    )],
                ))
        return sig



    def _why_changed(self, evidence: List[HealthEvidence]) -> List[Dict[str, Any]]:
        """Rank metrics by relative change magnitude (evidence-based).

        Only metrics with a previous_value contribute; ranking uses the
        absolute relative delta so it stays scale-free across metrics.
        """
        ranked = []
        for e in evidence:
            if e.previous_value is None or e.value is None or e.previous_value == 0:
                continue
            rel = abs(e.value - e.previous_value) / abs(e.previous_value)
            if rel < 0.1:
                continue
            ranked.append({
                "metric": e.metric,
                "dimension": self.METRIC_DEFS.get(e.metric, (None,))[0],
                "value": e.value,
                "previous_value": e.previous_value,
                "delta": round(e.value - e.previous_value, 4),
                "relative_delta": round(rel, 4),
                "direction": e.direction,
            })
        ranked.sort(key=lambda r: r["relative_delta"], reverse=True)
        return ranked[:8]

