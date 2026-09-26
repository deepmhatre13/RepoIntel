"""Project Health orchestration service (Stages 1-9, 22-29).

The single entry point used by both the API and a future monitoring
scheduler. It wires the pipeline together:

    normalize input -> GitHubHealthProvider.fetch -> feature engineering
        -> temporal analysis (trends/anomalies/change-points)
        -> HealthAnalysisEngine (dimensions/risks/correlation)
        -> persist observations -> LangGraph agent + Gemini -> report

It also supports an incremental ``refresh_project_health(repository)`` path
that fetches only new GitHub data and recomputes health features without
re-parsing a repository source tree (the monitoring-ready design).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from packages.project_health.models import (
    AnomalyResult,
    ChangePointResult,
    HealthAnalysis,
    HealthDimension,
    HealthState,
)
from packages.project_health.github_provider import (
    GitHubHealthProvider,
    RateLimitError,
    RepositoryNotFoundError,
    normalize_repository_input,
)
from packages.project_health.features import ProjectHealthFeatures
from packages.project_health.analysis import HealthAnalysisEngine
from packages.project_health import persistence, temporal


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _obs(ts: datetime, metric: str, value: Optional[float], *,
         source: str = "github", window: Optional[str] = None,
         availability: str = "AVAILABLE", confidence: float = 1.0) -> Dict[str, Any]:
    return {"timestamp": ts, "metric": metric, "value": value, "source": source,
            "observation_window": window, "availability": availability,
            "confidence": confidence}


def _dt(v):
    """Parse a datetime from a datetime-or-string (aware-safe)."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _find_metric(features: Dict[str, Any], metric: str) -> Optional[float]:
    """First real numeric value of a metric in the feature snapshot."""
    for group in features.values():
        if isinstance(group, dict) and metric in group:
            v = group[metric]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return None


def _slice_to_prior(data, end: datetime, start_days: int):
    """Deep-copy ``data`` restricted to events in [end - days, end]."""
    from copy import deepcopy
    from packages.project_health.models import ProviderData
    start = end - timedelta(days=start_days)

    def after(dt):
        dt = _dt(dt)
        return dt is not None and start <= dt <= end

    d = deepcopy(data)
    d.commits = [c for c in d.commits if after(c.committed_at)]
    d.pull_requests = [p for p in d.pull_requests
                       if after(p.opened_at) or after(p.closed_at) or after(p.merged_at)]
    d.issues = [i for i in d.issues
                if after(i.opened_at) or after(i.closed_at)]
    d.releases = [r for r in d.releases if after(r.published_at)]
    d.contributors = [c for c in data.contributors
                      if after(c.last_contribution) or after(c.first_contribution)]
    return d


def _slice_window(data, end: datetime, start_days: int):
    """Restrict ``data`` to its most recent ``start_days`` of activity."""
    return _slice_to_prior(data, end, start_days)


class ProjectHealthService:
    """Wires provider -> features -> temporal -> health engine -> agent."""

    def __init__(self, provider: Optional[GitHubHealthProvider] = None,
                 llm_client=None, window_days: int = 365):
        self.provider = provider or GitHubHealthProvider()
        self.llm = llm_client
        self.window_days = window_days
        self.engine = HealthAnalysisEngine()

    # -- observation building ------------------------------------------------
    def _build_observations(self, features: Dict[str, Any],
                            ts: datetime) -> List[Dict[str, Any]]:
        """Map feature snapshots into normalized historical observations.

        Only real numeric values are persisted as ``AVAILABLE``; metrics with
        no data are recorded with an explicit NOT_AVAILABLE availability so
        they are never confused with a measured zero.
        """
        obs: List[Dict[str, Any]] = []
        for group, f in features.items():
            if not isinstance(f, dict):
                continue
            metric = f.get("metric", group)
            for key, val in f.items():
                if key in ("metric", "values", "n", "valid", "distribution"):
                    continue
                if key.endswith("availability"):
                    if val == "NOT_AVAILABLE":
                        obs.append(_obs(ts, f"{group}_{key.split('_')[0]}", None,
                                        availability="NOT_AVAILABLE"))
                    continue
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    obs.append(_obs(ts, key, round(float(val), 4),
                                    window=f"{self.window_days}d"))
        return obs

    # -- core pipeline -------------------------------------------------------
    async def analyze_repository(self, owner: str, repo: str,
                                 question: str = "",
                                 persist: bool = True,
                                 include_source: bool = True,
                                 prior: Optional[Dict[str, Any]] = None) -> HealthAnalysis:
        """Run the full Project Health pipeline for a GitHub repository.

        When ``include_source`` is True (default), the pipeline acquires
        source intelligence (dependencies, security, code health) by safely
        cloning the repository and running the existing RepoIntel parsing
        pipeline. This makes Project Health standalone — the user does NOT
        need to first run Repository Intelligence.
        """
        analysis_id = str(uuid.uuid4())
        health_key = f"{owner}/{repo}".lower()
        try:
            data = self.provider.fetch_project_health_data(owner, repo)
        except RepositoryNotFoundError as exc:
            raise RepositoryNotFoundError(f"{owner}/{repo}: {exc}")
        except Exception as exc:
            raise RuntimeError(f"GitHub provider failed for {owner}/{repo}: {exc}")

        # Acquire source intelligence if requested (standalone workflow)
        source_intel = None
        if include_source:
            try:
                from packages.project_health.source_intelligence import acquire_source_intelligence
                source_intel = acquire_source_intelligence(owner, repo)
            except Exception as exc:
                import logging
                logging.getLogger("repointel.project_health").warning(
                    "Source intelligence acquisition failed for %s/%s: %s",
                    owner, repo, exc,
                )

        ts = _now()
        featurizer = ProjectHealthFeatures(data, source_intelligence=source_intel)
        features = featurizer.all_features()
        trends = self._compute_trends(features)
        anomalies, change_points = self._temporal_results(features)
        window_days = min(self.window_days, 90)
        changes = self._windowed_changes(data, window_days)
        correlations = await self._real_correlations(health_key, window_days)
        engine_out = self.engine.run(
            features=features, changes=changes, trends=trends,
            correlations=correlations,
            anomalies=anomalies, change_points=change_points, status=data.status,
        )
        engine_out["dimensions"] = self._with_dimension_trends(engine_out, changes)
        engine_out["changes"] = changes
        engine_out["correlations"] = correlations
        # Why-did-health-change is evidence-based, ranked by relative magnitude.
        engine_out["why_changed"] = sorted(
            [c for c in changes],
            key=lambda c: c.get("relative_delta", 0.0), reverse=True,
        )[:8]
        self._seed_evidence_baselines(engine_out["dimensions"], changes)
        analysis = self._assemble(analysis_id, data, engine_out, trends,
                                  anomalies, change_points, ts)
        # Monitoring refresh: emit evidence-backed health events vs the prior snapshot.
        if prior:
            analysis.health_events = self._derive_health_events(prior, analysis)


        # Include source intelligence availability in the analysis
        if source_intel:
            analysis.data_sufficiency["source_analysis"] = source_intel.availability
            analysis.data_sufficiency["source_details"] = {
                "dependencies": source_intel.dependencies.availability,
                "security": source_intel.security.status,
                "code_health": source_intel.code_health.availability,
            }

        if persist:
            await self._persist(health_key, owner, repo, data, features, analysis, ts)

        if self.llm is not None:
            analysis = await self._run_agent(analysis, question)

        return analysis

    async def _persist(self, health_key: str, owner: str, repo: str, data,
                       features: Dict[str, Any], analysis, ts: datetime) -> None:
        """Best-effort persistence of observations (monitoring foundation)."""
        try:
            observations = self._build_observations(features, ts)
            async with persistence.persistence_session() as session:
                meta = data.metadata
                await persistence.upsert_repository(
                    session, health_key=health_key, owner=owner, name=repo,
                    full_name=meta.full_name if meta else f"{owner}/{repo}",
                    url=meta.url if meta else "",
                    metadata_json=(json.dumps(meta.model_dump(mode="json", exclude_none=True))
                                   if meta else "{}"),
                )
                await persistence.add_observations(session, health_key, observations)
                await persistence.save_analysis(
                    session, health_key,
                    json.dumps(analysis.model_dump(mode="json")),
                )
                await persistence.touch_analysis(session, health_key)
        except Exception as exc:  # Persistence is best-effort for a read-only report.
            import logging
            logging.getLogger("repointel.project_health").warning(
                "health observation persistence failed for %s/%s: %s",
                owner, repo, exc,
            )

    # -- windowed temporal analysis -----------------------------------------
    def _windowed_changes(self, data, window_days: int) -> List[Dict[str, Any]]:
        """Compare recent vs prior window features -> evidence-based change list.

        Every entry carries value / previous_value / delta / direction so the
        engine's "why did health change" and dimension-trend logic consume real
        baselines instead of NOT_AVAILABLE.
        """
        from packages.project_health.models import TrendDirection
        end = _now()
        recent_f = ProjectHealthFeatures(_slice_window(data, end, window_days)).all_features()
        prior_f = ProjectHealthFeatures(
            _slice_to_prior(data, end - timedelta(days=window_days), window_days)
        ).all_features()

        changes: List[Dict[str, Any]] = []
        for metric in ("rolling_7d", "rolling_30d", "merge_rate",
                       "median_merge_days", "median_resolution_days",
                       "closure_rate", "stale_open_ratio", "stale_issue_ratio",
                       "median_interval_days", "frequency_per_90d",
                       "hhi", "contributor_count", "active_contributors_30d"):
            cur = _find_metric(recent_f, metric)
            prev = _find_metric(prior_f, metric)
            if cur is None or prev is None or prev == 0:
                continue
            rel = abs(cur - prev) / abs(prev)
            if rel < 0.1:
                continue
            dimension = self.engine.METRIC_DEFS.get(metric, (None,))[0]
            changes.append({
                "metric": metric, "dimension": dimension,
                "value": round(cur, 4), "previous_value": round(prev, 4),
                "delta": round(cur - prev, 4), "relative_delta": round(rel, 4),
                "direction": TrendDirection.INCREASING.value if cur > prev
                             else TrendDirection.DECREASING.value,
                "window": f"{window_days}d", "source": "github_metadata",
            })
        return changes

    @staticmethod
    def _seed_evidence_baselines(dimensions, changes):
        """Attach prior-window baselines to dimension evidence (drill-down).

        Each changed metric's prior value / delta / direction are written back
        onto the matching dimension evidence so the UI can show real
        ``value -> previous_value`` movement instead of a bare current value.
        """
        by_metric = {c["metric"]: c for c in changes}
        for d in dimensions:
            for e in d.evidence:
                c = by_metric.get(e.metric)
                if c:
                    e.previous_value = c.get("previous_value")
                    e.delta = c.get("delta")
                    e.direction = c.get("direction")

    def _with_dimension_trends(self, engine_out, changes):
        """Fold recent-vs-prior metric deltas into dimension-level trends."""
        from packages.project_health.models import HealthDimension, TrendDirection
        votes: Dict[str, List[int]] = {}
        for c in changes:
            dim = c.get("dimension")
            if not dim:
                continue
            higher = self.engine.METRIC_DEFS[c["metric"]][1]
            worsened = (c["delta"] > 0) if not higher else (c["delta"] < 0)
            bucket = votes.setdefault(dim, [0, 0])
            bucket[0 if worsened else 1] += 1
        out = []
        for d in engine_out["dimensions"]:
            v = votes.get(d.name, [0, 0])
            if sum(v) == 0:
                out.append(d)
                continue
            ratio = v[1] / sum(v)
            trend = (TrendDirection.INCREASING if ratio >= 0.65 else
                     TrendDirection.DECREASING if ratio <= 0.35 else
                     TrendDirection.STABLE)
            out.append(HealthDimension(**{**d.dict(), "trend": trend}))
        return out

    MIN_CORRELATION_OBSERVATIONS = 5

    async def _real_correlations(self, health_key: str, window_days: int):
        """Cross-time correlation from persisted per-snapshot observations.

        Phase H: correlation requires a real joint time series. This method
        reads the observation history accumulated across health snapshots
        (monitoring persistence). With fewer than
        ``MIN_CORRELATION_OBSERVATIONS`` (5, matching the change-point
        minimum) paired observations per metric, the result is reported as
        INSUFFICIENT_DATA — computing r over 2-3 points would yield
        meaningless ±1 values.
        """
        from packages.project_health.models import CorrelationResult, DataAvailability
        hist: Dict[str, List[float]] = {}
        try:
            async with persistence.persistence_session() as session:
                history = await persistence.get_all_metric_history(session, health_key)
            for metric, observations in (history.get("metrics") or {}).items():
                values = [o.get("value") for o in observations
                          if isinstance(o.get("value"), (int, float))
                          and o.get("availability", "AVAILABLE") == "AVAILABLE"]
                if values:
                    hist[metric] = [float(v) for v in values]
        except Exception:
            hist = {}

        pairs = [("hhi", "median_merge_days"),
                 ("hhi", "median_resolution_days"),
                 ("rolling_30d", "active_contributors_30d"),
                 ("rolling_30d", "frequency_per_90d"),
                 ("median_interval_days", "hhi")]
        out: List[CorrelationResult] = []
        for a, b in pairs:
            va, vb = hist.get(a), hist.get(b)
            n = min(len(va or []), len(vb or []))
            if n < self.MIN_CORRELATION_OBSERVATIONS:
                out.append(CorrelationResult(
                    metric_a=a, metric_b=b,
                    availability=DataAvailability.INSUFFICIENT_DATA.value,
                    sample_size=n,
                    note=(f"Only {n} paired observation(s) across snapshots; "
                          f"need at least {self.MIN_CORRELATION_OBSERVATIONS} "
                          "for a meaningful correlation."),
                ))
                continue
            r = temporal.pearson_correlation(va[-n:], vb[-n:])
            if r is None:
                out.append(CorrelationResult(
                    metric_a=a, metric_b=b,
                    availability=DataAvailability.INSUFFICIENT_DATA.value,
                    sample_size=n,
                    note="Zero variance in one series over the window.",
                ))
                continue
            out.append(CorrelationResult(
                metric_a=a, metric_b=b, correlation_coefficient=round(r, 3),
                observation_window=f"{window_days}d", sample_size=n,
                confidence=round(min(1.0, n / 12.0), 3),
                availability=DataAvailability.AVAILABLE.value,
                note="Signals moved together over the window (correlation, not causation).",
            ))
        return out

    def _compute_trends(self, features: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Trend over the real numeric daily commit series (Phase H).

        The daily per-day commit counts are numeric; the observation index is
        the x-axis (uniform daily spacing). Datetime lists are not numeric and
        previously produced an empty array -> permanent NOT_AVAILABLE trends.
        """
        trends: Dict[str, Dict[str, Any]] = {}
        activity = features.get("activity") or {}
        daily = activity.get("daily_counts") or []
        if daily:
            trend = temporal.estimate_trend(list(range(len(daily))), daily)
            trend["metric"] = "commits"
            trends["commits"] = trend
        else:
            trends["commits"] = {
                "slope": None, "direction": "NOT_AVAILABLE", "n": 0,
                "r_value": None, "p_value": None,
                "reason": "no timestamped commits in window",
            }
        return trends

    def _temporal_results(self, features: Dict[str, Any]):
        """Anomaly + change-point detection over the numeric daily series.

        Timestamps are derived from the series start date reported by the
        featurizer (real dates, not fabricated ones).
        """
        anomalies: List[AnomalyResult] = []
        change_points: List[ChangePointResult] = []
        activity = features.get("activity") or {}
        daily = activity.get("daily_counts") or []
        if daily:
            start_str = activity.get("daily_counts_start")
            try:
                start_day = datetime.fromisoformat(str(start_str)).date()
            except (TypeError, ValueError):
                start_day = None
            if start_day is not None:
                ts_slots = [datetime(start_day.year, start_day.month, start_day.day,
                                     tzinfo=timezone.utc) + timedelta(days=i)
                            for i in range(len(daily))]
            else:
                ts_slots = list(range(len(daily)))
            for a in temporal.detect_anomalies("commits", ts_slots, daily):
                anomalies.append(AnomalyResult(**a))
            for cp in temporal.detect_change_points(daily):
                idx = cp.get("index")
                change_points.append(ChangePointResult(
                    metric="commits", index=idx,
                    timestamp=(ts_slots[idx].isoformat()
                               if idx is not None and idx < len(ts_slots) else None),
                    before_mean=cp["before_mean"], after_mean=cp["after_mean"],
                    before_count=cp["before_count"], after_count=cp["after_count"],
                    shift_magnitude=cp["shift_magnitude"],
                    description="Commits regime shifted significantly.",
                ))
        return anomalies, change_points

    def _assemble(self, analysis_id: str, data, engine_out, trends,
                  anomalies, change_points, ts) -> HealthAnalysis:
        meta = data.metadata
        owner = meta.owner if meta else ""
        repo = meta.name if meta else ""
        full_name = meta.full_name if meta else ""
        url = meta.url if meta else ""
        dims = engine_out["dimensions"]
        trajectory, traj_conf = self._derive_trajectory(dims)
        score_state, missing = self._score_state(engine_out, dims)
        return HealthAnalysis(
            analysis_id=analysis_id,
            owner=owner, repository=repo, full_name=full_name,
            url=url, created_at=ts, availability="AVAILABLE",
            overall_score=engine_out["overall_score"],
            score_state=score_state,
            missing_dimensions=missing,
            score_breakdown=engine_out.get("score_breakdown") or {},
            trajectory=trajectory, trajectory_confidence=traj_conf,
            dimensions=dims,
            risks=engine_out["risks"],
            signals=engine_out["signals"],
            correlations=engine_out["correlations"],
            anomalies=anomalies,
            change_points=change_points,
            trends=list(trends.values()),
            timeline=self._build_timeline(engine_out["changes"], dims),
            why_changed=engine_out.get("why_changed") or [],
            changes=engine_out.get("changes") or [],
            contributors=data.contributors,
            releases=data.releases,
            repository_metadata=meta,
            data_sufficiency=self._sufficiency(engine_out),
        )

    @staticmethod
    def _score_state(engine_out: Dict[str, Any], dims: List[HealthDimension]) -> Tuple[str, List[str]]:
        """Deterministic explainable confidence state for the overall score."""

        from packages.project_health.models import ScoreState
        scored = [d for d in dims if d.score is not None]
        total = len(dims)
        if not scored or total == 0:
            return ScoreState.NOT_SCORABLE.value, [d.name for d in dims]
        coverage = len(scored) / total
        avg_conf = sum(d.confidence for d in scored) / len(scored)
        # A LOW_CONFIDENCE score is reported explicitly — never a fake 0-100.

        if coverage >=   0.8 and avg_conf >=  0.45:
            return ScoreState.COMPUTED.value, [d.name for d in dims if d.score is None]
        if coverage >=  0.5:
            return ScoreState.LOW_CONFIDENCE.value, [d.name for d in dims if d.score is None]
        return ScoreState.INSUFFICIENT_DATA.value, [d.name for d in dims if d.score is None]

    @staticmethod
    def _derive_trajectory(dims: List[HealthDimension]):
        """Overall trajectory from dimension-level trends (decomposable).

        Health semantics: a dimension's INCREASING trend means its component
        metrics moved in their favorable direction (each metric's direction is
        already resolved against its higher_is_better bound), DECREASING means
        they deteriorated. Requires >= 2 directional dimensions before a label
        is issued; otherwise the state is explicitly INSUFFICIENT_DATA.
        """
        from packages.project_health.models import TrendDirection
        improving = sum(1 for d in dims if d.trend == TrendDirection.INCREASING)
        deteriorating = sum(1 for d in dims if d.trend == TrendDirection.DECREASING)
        scored = improving + deteriorating
        if scored < 2:
            return "INSUFFICIENT_DATA", 0.0
        ratio = improving / scored
        conf = round(min(1.0, scored / 4.0), 3)
        if improving >= 2 and deteriorating >= 2:
            return "VOLATILE", conf
        if ratio >= 0.65:
            return "IMPROVING", conf
        if ratio <= 0.35:
            return "DETERIORATING", conf
        return "STABLE", conf

    def _build_timeline(self, changes: List[Dict[str, Any]],
                        dims: List[HealthDimension]) -> List[Dict[str, Any]]:
        """Derived project timeline from real signals (no LLM invention)."""
        events = []
        now = _now()
        for d in dims:
            if d.state in (HealthState.DETERIORATING, HealthState.IMPROVING):
                events.append({
                    "timestamp": now.isoformat(),
                    "event": f"{d.name} {d.state.value.lower()}",
                    "dimension": d.name,
                    "metrics": d.supporting_metrics[:4],
                })
        for c in changes:
            events.append({
                "timestamp": now.isoformat(),
                "event": f"{c.get('metric')} changed by {c.get('delta')}",
                "dimension": c.get("dimension"),
                "metrics": [c.get("metric")],
            })
        return events

    def _sufficiency(self, engine_out: Dict[str, Any]) -> Dict[str, Any]:
        dims = engine_out.get("dimensions") or []
        scored = [d for d in dims if d.score is not None]
        score_state, missing = self._score_state(engine_out, dims)
        return {
            "dimensions_computed": len(scored),
            "dimensions_total": len(self.engine.DIMENSION_NAMES),
            "coverage": engine_out.get("score_coverage", 0.0),
            "observation_window": f"{self.window_days}d",
            "score_state": score_state,
            "missing_dimensions": missing,
            "score_breakdown": engine_out.get("score_breakdown") or {},
        }

    def _derive_health_events(self, prior, current):
        """Emit evidence-backed health events comparing prior vs current.

        Deterministic: only state transitions emit events; the LLM is never
        involved. Evidence is taken directly from the current dimensions.
        """
        from packages.project_health.models import HealthEvent, HealthState

        events = []
        now = _now().isoformat()
        prior_dims = {d.get("name"): d for d in (prior or {}).get("dimensions") or []}
        for d in current.dimensions:
            if d.score is None:
                continue
            pd = prior_dims.get(d.name)
            if not pd:
                continue
            prev = pd.get("state")
            cur = d.state.value
            if not prev or prev == cur:
                continue
            if cur == HealthState.DETERIORATING.value:
                kind, sev = "HEALTH_DETERIORATED", "warning"
            elif cur == HealthState.IMPROVING.value or (
                    cur == HealthState.GOOD.value
                    and prev in (HealthState.DETERIORATING.value, HealthState.STABLE.value)):
                kind, sev = "HEALTH_IMPROVED", "info"
            else:
                continue
            events.append(HealthEvent(
                kind=kind,
                dimension=d.name,
                severity=sev,
                summary="{0} health went {1} -> {2} (statistical trend, not causation).".format(
                    d.name, prev.lower(), cur),
                evidence=list(d.evidence or [])[:4],
                observed_at=now,
            ))
        prev_traj = (prior or {}).get("trajectory")
        if prev_traj and current.trajectory and prev_traj != current.trajectory:
            events.append(HealthEvent(
                kind="HEALTH_TRAJECTORY_CHANGED",
                dimension="overall",
                severity="info",
                summary="Overall trajectory went {0} -> {1} (statistical, not causal).".format(
                    prev_traj, current.trajectory),
                evidence=[],
                observed_at=now,
            ))
        return events


    async def _run_agent(self, analysis: HealthAnalysis,
                         question: str) -> HealthAnalysis:
        from packages.project_health.agent import ProjectHealthAgent
        agent = ProjectHealthAgent(self.llm)
        result = await agent.analyze(analysis, question or "Overall project health")
        analysis.gemini_analysis = result.get("gemini_interpretation")
        analysis.ai_analysis = analysis.gemini_analysis
        analysis.ai_available = bool(analysis.gemini_analysis)
        analysis.gemini_available = analysis.ai_available
        analysis.reviewed = True
        analysis.review_notes = result.get("review_notes") or []
        return analysis

    async def refresh_project_health(self, repository: str) -> HealthAnalysis:
        """Incremental refresh entry point (monitoring boundary).

        Loads the last persisted snapshot as ``prior``, fetches fresh GitHub
        metadata, and re-uses the existing source snapshot only when the
        repository's ``pushed_at`` fingerprint changed -- so unchanged source is
        never re-parsed. Re-computes health, appends observations, and emits
        evidence-backed health events describing dimension transitions versus the
        prior snapshot (deterministic -- no LLM involvement).
        """
        owner_repo = normalize_repository_input(repository)
        owner, repo = owner_repo.split("/", 1)
        health_key = owner_repo.lower()

        prior = None
        try:
            async with persistence.persistence_session() as session:
                raw = await persistence.load_analysis(session, health_key)
                prior = json.loads(raw) if raw else None
        except Exception:
            prior = None

        include_source = True
        try:
            meta = self.provider.fetch_repository(owner, repo)
            prior_meta = (prior or {}).get("repository_metadata") or {}
            prior_pushed = prior_meta.get("pushed_at")
            curr_pushed = meta.pushed_at.isoformat() if meta and meta.pushed_at else None
            if prior_pushed and curr_pushed and prior_pushed == curr_pushed:
                include_source = False
        except RateLimitError as exc:
            # Fail loudly rather than overwrite a good snapshot with a
            # degraded one: monitoring must never silently lose evidence.
            raise RuntimeError(
                f"GitHub API rate limit exceeded; refresh aborted for "
                f"{health_key} ({exc})")
        except Exception:
            include_source = True

        analysis = await self.analyze_repository(
            owner, repo, question="Refresh project health",
            include_source=include_source, prior=prior)

        if analysis.health_events:
            try:
                async with persistence.persistence_session() as session:
                    await persistence.save_analysis(
                        session, health_key,
                        json.dumps(analysis.model_dump(mode="json")))
                    await persistence.touch_analysis(session, health_key)
            except Exception:
                pass
        return analysis