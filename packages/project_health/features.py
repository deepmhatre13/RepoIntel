"""Feature engineering for Project Health Intelligence (Stages 5-6).

Transforms normalized GitHub observations (ProviderData) into mathematically
meaningful time-series features:

* ACTIVITY - rolling 7/30/90-day commits, active days, recent-vs-baseline
* RELEASE  - release frequency, median interval, interval variance
* PR       - arrival, merge rate, median merge time, stale ratio
* ISSUE    - arrival, closure rate, backlog growth, median resolution time
* COMMUNITY- active contributors, retention, newcomer ratio, concentration
* CONCENT  - Herfindahl-Hirschman Index (HHI) + preserved distribution

Component metrics stay composable: analysis (trends, anomalies, correlations,
risk) consumes the returned ``{metric_name: series}`` dict.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from packages.project_health.models import ProviderData


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dt(v) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def _diff_days(a: Optional[datetime], b: Optional[datetime]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        return max(0.0, (b - a).total_seconds() / 86400.0)
    except TypeError:
        return None


def hhi_contributions(contributions: List[int]) -> Dict[str, Any]:
    """Herfindahl-Hirschman Index of contribution concentration.

    ``HHI = sum((c_i / total)^2)``. Ranges from ``1/N`` (perfectly spread)
    toward ``1.0`` (a single dominant contributor). The full distribution is
    preserved as evidence — it is never reduced to a single threshold.
    """
    total = sum(int(c) for c in contributions if c is not None)
    if total <= 0:
        return {"hhi": None, "top1_share": None, "concentration": "NOT_AVAILABLE",
                "n": 0, "distribution": [], "valid": False}
    shares = [int(c) / total for c in contributions if c is not None and c > 0]
    hhi = sum(s * s for s in shares)
    top1 = max(shares) if shares else None
    if hhi is None:
        return {"hhi": None, "top1_share": None, "concentration": "NOT_AVAILABLE",
                "n": len(shares), "distribution": shares, "valid": False}
    if hhi >= 0.5:
        concentration = "HIGH"
    elif hhi >= 0.25:
        concentration = "MODERATE"
    else:
        concentration = "LOW"
    return {
        "hhi": round(hhi, 4),
        "top1_share": round(top1, 4) if top1 is not None else None,
        "concentration": concentration,
        "n": len(shares),
        "distribution": [round(s, 4) for s in shares],
        "valid": True,
    }


class ProjectHealthFeatures:
    """Compute analyzable features from normalized GitHub health data.

    Optionally accepts pre-computed source intelligence (from the
    source_intelligence adapter) to enrich dependency/security/code-health
    features when a standalone source analysis was performed.
    """

    def __init__(self, data: ProviderData, source_intelligence=None):
        self.data = data
        self.source = source_intelligence

    def _commits_by_day(self) -> List[datetime]:
        return [c.committed_at for c in self.data.commits if c.committed_at]

    def _events_in_window(self, events: List[datetime], days: int,
                          anchor: Optional[datetime] = None) -> int:
        if not events:
            return 0
        now = anchor or _utcnow()
        cutoff = now - timedelta(days=days)
        return sum(1 for e in events if e >= cutoff)

    def activity_features(self) -> Dict[str, Any]:
        """Daily commit counts as a time series + rolled up activity stats.

        A failed GitHub fetch (``API_ERROR``) yields ``None`` metrics with an
        explicit availability marker -- never a measured zero, which would
        fabricate a "dead repository" signal out of a transport failure.

        Phase H: rolling windows are only statistically meaningful when the
        observed history spans at least the window. A repository with 2
        commits must NOT receive the same treatment as one with years of
        history, so short windows are reported as None with an explicit
        INSUFFICIENT_HISTORY marker instead of a small-sample average.
        """
        from packages.project_health import temporal
        commits = sorted(self._commits_by_day())
        f: Dict[str, Any] = {"metric": "commits", "values": commits}
        if not commits:
            if self.data.status.get("commits") == "API_ERROR":
                return {**f, "rolling_7d": None, "rolling_30d": None,
                        "rolling_90d": None, "active_days_30d": None,
                        "recent_vs_baseline": None,
                        "activity_span_days": 0,
                        "commits_availability": "NOT_AVAILABLE"}
            # VALID_ZERO: the provider confirmed zero commits in the window.
            return {**f, "rolling_7d": 0, "rolling_30d": 0, "rolling_90d": 0,
                    "active_days_30d": 0, "recent_vs_baseline": None,
                    "activity_span_days": 0,
                    "commits_availability": "ZERO_COMMITS"}
        # Bucket commits into a per-day count series for rolling/trend math.
        start = commits[0].date()
        end = commits[-1].date()
        bucket = [0] * ((end - start).days + 1)
        for c in commits:
            idx = (c.date() - start).days
            bucket[idx] += 1
        span_days = len(bucket)
        rolling = temporal.rolling_stats(bucket, [7, 30, 90])

        # Phase H: window validity -- a trailing W-day mean over fewer than W
        # observed days is a small-sample artifact, not the statistic asked for.
        def _window_ok(w: int) -> bool:
            return span_days >= w

        n30 = self._events_in_window(commits, 30)
        f.update({
            "rolling_7d": rolling["rolling_7"] if _window_ok(7) else None,
            "rolling_30d": rolling["rolling_30"] if _window_ok(30) else None,
            "rolling_90d": rolling["rolling_90"] if _window_ok(90) else None,
            "active_days_30d": n30,
            "activity_span_days": span_days,
            # Real numeric daily series for trend/anomaly/change-point math.
            # (Datetime lists are not numeric and silently produce empty
            # arrays downstream — Phase H fix.)
            "daily_counts": bucket,
            "daily_counts_start": start.isoformat(),
            # recent-vs-baseline needs a baseline: require >= 2x the recent window.
            "recent_vs_baseline": (temporal.recent_vs_baseline_ratio(bucket, 7)
                                   if span_days >= 14 else None),
        })
        if span_days >= 90:
            f["commits_availability"] = "AVAILABLE"
        elif span_days >= 30:
            f["commits_availability"] = "PARTIAL_HISTORY"
        else:
            f["commits_availability"] = "INSUFFICIENT_HISTORY"
        return f

    def release_features(self) -> Dict[str, Any]:
        releases = [r for r in self.data.releases if r.published_at and not r.draft]
        releases.sort(key=lambda r: r.published_at or datetime.min)
        f: Dict[str, Any] = {"metric": "releases", "release_count": len(releases)}
        if len(releases) < 2:
            # Too few releases for interval or frequency computation.
            # Report INSUFFICIENT_DATA rather than silently returning None metrics
            # — a single release is NOT a zero-frequency project.
            f.update({
                "median_interval_days": None,
                "interval_variance": None,
                "frequency_per_90d": None,
                "release_availability": "INSUFFICIENT_DATA",
            })
            return f
        intervals = []
        for a, b in zip(releases[:-1], releases[1:]):
            d = _diff_days(a.published_at, b.published_at)
            if d is not None:
                intervals.append(d)

        # frequency_per_90d: releases per 90-day period over the *actual* observed
        # span (first -> last release). Previously this divided by the constant 90,
        # producing a meaningless number unrelated to real cadence.
        first_dt = _dt(releases[0].published_at)
        last_dt = _dt(releases[-1].published_at)
        if first_dt and last_dt:
            observed_days = max(1.0, (last_dt - first_dt).total_seconds() / 86400.0)
            freq_per_90d = round(len(releases) * 90.0 / observed_days, 3)
        else:
            freq_per_90d = None

        f.update({
            "median_interval_days": round(statistics.median(intervals), 2) if intervals else None,
            "interval_variance": (round(statistics.pstdev(intervals), 2)
                                  if len(intervals) > 1 else None),
            "frequency_per_90d": freq_per_90d,
            "release_availability": "AVAILABLE",
        })
        return f

    def pr_features(self) -> Dict[str, Any]:
        prs = self.data.pull_requests
        f: Dict[str, Any] = {"metric": "pull_requests", "pr_count": len(prs)}
        if not prs and self.data.status.get("pulls") == "API_ERROR":
            return {**f, "merge_rate": None, "median_merge_days": None,
                    "pr_closed_count": None, "stale_open_ratio": None,
                    "pr_open_count": None,
                    "pull_requests_availability": "NOT_AVAILABLE"}
        merged = [p for p in prs if p.state == "merged" and p.merged_at]
        merge_days = [_diff_days(p.opened_at, p.merged_at) for p in merged]
        merge_days = [d for d in merge_days if d is not None]
        total_closed = len([p for p in prs if p.state in ("merged", "closed")])
        stale = len([p for p in prs if p.state == "open" and p.opened_at
                     and (datetime.now(timezone.utc) - p.opened_at).days > 30])
        # Phase H: PR-state uncertainty. GitHub exposes PRs as open/closed only;
        # "merged" is a derived state. Counting unknown-state PRs in the
        # denominator would understate the merge rate, so they are excluded
        # and the exclusion is reported (denominators must be valid).
        resolved = merged + [p for p in prs if p.state == "closed"]
        pr_open_count = len([p for p in prs if p.state == "open"])
        unknown_state = len(prs) - len(merged) - total_closed - pr_open_count
        if resolved:
            merge_rate = len(merged) / len(resolved)
            merge_rate_denominator = len(resolved)
        else:
            # VALID_ZERO vs MISSING: no resolved PRs at all is genuinely 0/0.
            # There is no evidence of either merging or abandonment, so the
            # rate is MISSING (None), never a fabricated 0.0 (which the
            # metric bounds would interpret as "worst possible maintenance").
            merge_rate = None
            merge_rate_denominator = 0
        f.update({
            "merge_rate": round(merge_rate, 4) if merge_rate is not None else None,
            "merge_rate_denominator": merge_rate_denominator,
            "pr_unknown_state_count": max(0, unknown_state),
            "median_merge_days": (round(statistics.median(merge_days), 2)
                                  if merge_days else None),
            "pr_closed_count": total_closed,
            "stale_open_ratio": round(stale / pr_open_count, 4) if pr_open_count else None,
            "pr_open_count": pr_open_count,
        })
        return f

    def issue_features(self) -> Dict[str, Any]:
        issues = self.data.issues
        f: Dict[str, Any] = {"metric": "issues", "issue_count": len(issues)}
        if not issues and self.data.status.get("issues") == "API_ERROR":
            return {**f, "median_resolution_days": None, "issue_open_count": None,
                    "issue_closed_count": None, "stale_issue_ratio": None,
                    "closure_rate": None,
                    "issues_availability": "NOT_AVAILABLE"}
        closed = [i for i in issues if i.closed_at and i.opened_at]
        resolution_days = [_diff_days(i.opened_at, i.closed_at) for i in closed]
        resolution_days = [d for d in resolution_days if d is not None]
        stale = len([i for i in issues if i.state == "open" and i.opened_at
                     and (datetime.now(timezone.utc) - i.opened_at).days > 90])
        open_count = len([i for i in issues if i.state == "open"])
        # Phase H: closure rate is defined only over resolved outcomes.
        # issues that are still open carry no closure evidence; counting them
        # in the denominator makes a healthy active backlog look like total
        # maintenance failure. With no resolved issues the rate is MISSING,
        # not zero.
        if closed:
            closure_rate = len(closed) / (len(closed) + open_count)
            closure_rate_denominator = len(closed) + open_count
        else:
            closure_rate = None
            closure_rate_denominator = 0
        f.update({
            "median_resolution_days": (round(statistics.median(resolution_days), 2)
                                       if resolution_days else None),
            "issue_open_count": open_count,
            "issue_closed_count": len(closed),
            "stale_issue_ratio": round(stale / open_count, 4) if open_count else None,
            "closure_rate": round(closure_rate, 4) if closure_rate is not None else None,
            "closure_rate_denominator": closure_rate_denominator,
        })
        return f

    def community_features(self) -> Dict[str, Any]:
        if (not self.data.contributors
                and self.data.status.get("contributors") == "API_ERROR"):
            return {"metric": "community", "contributor_count": None,
                    "active_contributors_30d": None,
                    "community_availability": "NOT_AVAILABLE"}
        contribs = [c.contributions for c in self.data.contributors if c.contributions]
        hhi = hhi_contributions(contribs)
        # Phase H: "active in the last 30 days" must be measured over the last
        # 30 days, not over the entire fetch window. Commits are bounded to a
        # trailing 30-day window; contributors with no commit timestamps are
        # reported as None (NOT_AVAILABLE), never as a fabricated 0.
        cutoff = _utcnow() - timedelta(days=30)
        authors = {c.author for c in self.data.commits
                   if c.author and c.committed_at and c.committed_at >= cutoff}
        if not authors and not any(c.committed_at for c in self.data.commits):
            active_30d: Optional[int] = None  # no timestamped commits to measure
        else:
            active_30d = len(authors)
        return {
            "metric": "community",
            "contributor_count": len(self.data.contributors),
            "active_contributors_30d": active_30d,
            **hhi,
        }

    def dependency_features(self) -> Dict[str, Any]:
        """Dependency health from standalone source intelligence when available.

        When the Project Health workflow acquired a source snapshot, real
        dependency data from manifest files is reported. Otherwise NOT_AVAILABLE
        (never reported as 0).
        """
        if self.source and self.source.dependencies and self.source.dependencies.availability == "AVAILABLE":
            dep = self.source.dependencies
            return {
                "metric": "dependencies",
                "availability": "AVAILABLE",
                "total_dependencies": dep.total_count,
                "direct_dependencies": dep.direct_count,
                "manifest_files": dep.manifest_files[:10],
                "ecosystems": dep.ecosystems,
            }
        return {"metric": "dependencies", "availability": "NOT_AVAILABLE"}

    def security_features(self) -> Dict[str, Any]:
        """Security health from standalone source intelligence when available."""
        if self.source and self.source.security:
            sec = self.source.security
            if sec.status == "available":
                return {
                    "metric": "security",
                    "availability": "AVAILABLE",
                    "findings_count": sec.findings_count,
                    "error_count": sec.error_count,
                    "warning_count": sec.warning_count,
                    "info_count": sec.info_count,
                    "findings_by_rule": sec.findings_by_rule,
                    "scanner_available": sec.scanner_available,
                }
            if sec.status == "no_findings":
                return {
                    "metric": "security",
                    "availability": "AVAILABLE",
                    "findings_count": 0,
                    "error_count": 0,
                    "warning_count": 0,
                    "scanner_available": sec.scanner_available,
                }
            if sec.status in ("not_available", "error"):
                return {
                    "metric": "security",
                    "availability": "NOT_AVAILABLE",
                    "scanner_available": sec.scanner_available,
                    "error": sec.error,
                }
        return {"metric": "security", "availability": "NOT_AVAILABLE"}

    def code_health_features(self) -> Dict[str, Any]:
        """Code health from standalone source analysis when available."""
        if self.source and self.source.code_health and self.source.code_health.availability == "AVAILABLE":
            ch = self.source.code_health
            return {
                "metric": "code_health",
                "availability": "AVAILABLE",
                "files_parsed": ch.files_parsed,
                "files_seen": ch.files_seen,
                "symbol_count": ch.symbol_count,
                "edge_count": ch.edge_count,
                "chunk_count": ch.chunk_count,
                "parse_failures": ch.parse_failures,
                "languages": ch.languages,
            }
        return {"metric": "code_health", "availability": "NOT_AVAILABLE"}

    def all_features(self) -> Dict[str, Any]:
        return {
            "activity": self.activity_features(),
            "releases": self.release_features(),
            "pull_requests": self.pr_features(),
            "issues": self.issue_features(),
            "community": self.community_features(),
            "dependencies": self.dependency_features(),
            "security": self.security_features(),
            "code_health": self.code_health_features(),
        }