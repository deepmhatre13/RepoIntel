"""Temporal statistics for Project Health Intelligence.

Established statistical methods (numpy / scipy) applied to normalized health
observation time series. These are the reusable analytical primitives that
keep the health engine compact — no hand-rolled if/else scoring.

Current-state vs trend vs anomaly vs change-point vs forecast are distinct
concepts and are computed as such:

* ``robust_zscore`` / ``mad``            - scale-invariant anomaly detection
* ``detect_change_points``               - regime shift via mean-shift with
                                           MAD-based effect size
* ``estimate_trend``                     - robust slope (Theil-Sen when enough
                                           data, least squares otherwise)
* ``recent_vs_baseline_ratio``           - recent-vs-baseline activity
* ``rolling_stats``                      - 7/30/90-day rolling windows
* ``pearson_correlation`` / ``spearman`` - cross-metric movement alignment
* ``forecast_linear``                    - staged trend projection, clearly
                                           labelled forecast (never \"prediction\")
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as _stats
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


MIN_OBSERVATIONS_CHANGEPOINT = 5
MIN_OBSERVATIONS_TREND = 3


def _to_numeric(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [v for v in values if isinstance(v, (int, float)) and v is not None],
        dtype=float,
    )


def mad(values: Sequence[float]) -> float:
    """Median absolute deviation (robust spread estimator)."""
    arr = _to_numeric(values)
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def robust_zscore(value: float, baseline: Sequence[float]) -> float:
    """Robust (MAD-based) z-score of ``value`` relative to ``baseline``."""
    baseline_arr = _to_numeric(baseline)
    if baseline_arr.size == 0:
        return 0.0
    med = float(np.median(baseline_arr))
    spread = mad(baseline_arr)
    if spread <= 0.0:
        spread = float(np.std(baseline_arr)) if baseline_arr.size > 1 else 1e-12
    if spread <= 0.0:
        spread = 1e-12
    return float((value - med) / (1.4826 * spread))


def rolling_stats(values: Sequence[float], windows: Sequence[int]) -> Dict[str, Optional[float]]:
    """Trailing rolling mean for several windows; None when a window is empty."""
    arr = _to_numeric(values)
    out: Dict[str, Optional[float]] = {}
    for w in windows:
        out[f"rolling_{w}"] = (
            None if arr.size == 0
            else round(float(arr[-min(w, arr.size):].mean()), 4)
        )
    return out


def exponential_weighted_mean(values: Sequence[float], span: float = 90) -> Optional[float]:
    """Exponentially decaying average (recent-weighted central tendency)."""
    arr = _to_numeric(values)
    if arr.size == 0:
        return None
    alpha = 2.0 / (span + 1.0)
    out = float(arr[0])
    for v in arr[1:]:
        out = alpha * v + (1 - alpha) * out
    return round(out, 4)


def recent_vs_baseline_ratio(values: Sequence[float], recent_window: int = 7) -> Optional[float]:
    """Mean of last ``recent_window`` vs mean of the rest (recent-baseline ratio)."""
    arr = _to_numeric(values)
    if arr.size < 2:
        return None
    recent = arr[-recent_window:] if arr.size >= recent_window else arr
    baseline = arr[:-len(recent)]
    if baseline.size == 0 or float(np.mean(baseline)) == 0:
        return None
    return round(float(np.mean(recent)) / float(np.mean(baseline)), 4)


def estimate_trend(timestamps: Sequence[Any], values: Sequence[float]) -> Dict[str, Any]:
    """Estimate a linear trend (slope) over time.

    Uses scipy's Theil-Sen (robust) estimator when enough data exists;
    otherwise least squares via numpy. Returns slope / direction / r / p.
    """
    arr = _to_numeric(values)
    n = arr.size
    if n < MIN_OBSERVATIONS_TREND:
        return {"slope": None, "direction": "NOT_AVAILABLE", "n": n,
                "r_value": None, "p_value": None}
    x = np.arange(n, dtype=float)
    # A constant series has no trend by definition; correlating against it
    # would divide by zero (nan r). Report STABLE with no correlation.
    if float(np.std(arr)) == 0.0:
        return {"slope": 0.0, "direction": "STABLE", "n": n,
                "r_value": None, "p_value": None}
    try:
        if _HAS_SCIPY and n >= 8:
            res = _stats.theilslopes(arr, x)
            slope = float(res[0])
            r_value = float(np.corrcoef(x, arr)[0, 1])
            p_value = None
        else:
            slope = float(np.polyfit(x, arr, 1)[0])
            r_value = float(np.corrcoef(x, arr)[0, 1])
            _, p_value = _stats.pearsonr(x, arr) if _HAS_SCIPY else (None, None)
    except Exception:  # pragma: no cover
        return {"slope": None, "direction": "NOT_AVAILABLE", "n": n,
                "r_value": None, "p_value": None}
    if slope > 1e-9:
        direction = "INCREASING"
    elif slope < -1e-9:
        direction = "DECREASING"
    else:
        direction = "STABLE"
    return {"slope": slope, "direction": direction, "n": n,
            "r_value": r_value, "p_value": p_value}


def detect_change_points(values: Sequence[float]) -> List[Dict[str, Any]]:
    """Detect a meaningful mean regime shift (change-point detection).

    For each candidate split, compare before/after means weighed against the
    pooled spread (MAD). Reports a shift only when its effect size is
    meaningful (|d| >= 0.5, a common Cohen's d convention) — no noise alerts.
    """
    arr = _to_numeric(values)
    n = arr.size
    if n < MIN_OBSERVATIONS_CHANGEPOINT:
        return []
    best: Optional[Dict[str, Any]] = None
    for i in range(1, n):
        before = arr[:i]
        after = arr[i:]
        if before.size < 2 or after.size < 2:
            continue
        bmean = float(np.mean(before))
        amean = float(np.mean(after))
        effect = (amean - bmean) / (0.5 * (mad(before) + mad(after)) + 1e-9)
        cand = {
            "index": i,
            "before_mean": round(bmean, 4),
            "after_mean": round(amean, 4),
            "before_count": int(before.size),
            "after_count": int(after.size),
            "shift_magnitude": round(amean - bmean, 4),
            "effect_size": round(float(effect), 4),
        }
        if best is None or abs(effect) > abs(best["effect_size"]):
            best = cand
    if best is not None and abs(best["effect_size"]) >= 0.5:
        return [best]
    return []


def detect_anomalies(
    metric: str,
    timestamps: Sequence[Any],
    values: Sequence[float],
    threshold: float = 3.0,
) -> List[Dict[str, Any]]:
    """Flag robust-z outliers: ``anomaly`` vs ``persistent_change`` vs noise.

    A single extreme point is an anomaly; 3+ consecutive extreme points are a
    persistent change. No arbitrary one-off thresholds — robust z-scores over
    the MAD baseline only.
    """
    arr = _to_numeric(values)
    n = arr.size
    out: List[Dict[str, Any]] = []
    if n < 5:
        return out
    med = float(np.median(arr))
    spread = mad(arr) or (float(np.std(arr)) if n > 1 else 0.0)
    if spread <= 0.0:
        return out
    flags = [abs(float((v - med) / (1.4826 * spread))) >= threshold for v in arr]
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    for (start, end) in runs:
        is_persistent = (end - start + 1) >= 3
        out.append(
            {
                "metric": metric,
                "kind": "persistent_change" if is_persistent else "anomaly",
                "timestamp": str(timestamps[end]) if end < len(timestamps) else None,
                "robust_zscore": round(float((arr[end] - med) / (1.4826 * spread)), 4),
                "value": round(float(arr[end]), 4),
                "baseline_median": round(med, 4),
                "baseline_mad": round(float(spread), 4),
                "observation_count": n,
                "description": (
                    "Several consecutive points outside the robust baseline"
                    if is_persistent
                    else "Isolated value far outside the robust baseline"
                ),
            }
        )
    return out


def pearson_correlation(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Pearson correlation between two aligned numeric series (trailing)."""
    a_arr = _to_numeric(a)
    b_arr = _to_numeric(b)
    k = min(a_arr.size, b_arr.size)
    a_arr = a_arr[-k:]
    b_arr = b_arr[-k:]
    if k < 3 or float(np.std(a_arr)) == 0 or float(np.std(b_arr)) == 0:
        return None
    return float(np.corrcoef(a_arr, b_arr)[0, 1])


def spearman_correlation(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation (robust to monotonic non-linear linkage)."""
    a_arr = _to_numeric(a)
    b_arr = _to_numeric(b)
    k = min(a_arr.size, b_arr.size)
    if k < 4:
        return None
    if _HAS_SCIPY:
        out, _ = _stats.spearmanr(a_arr[-k:], b_arr[-k:])
        return float(out) if not np.isnan(out) else None
    return None


def forecast_linear(
    timestamps: Sequence[Any],
    values: Sequence[float],
    horizon: int = 30,
    min_history: int = 10,
) -> Dict[str, Any]:
    """Staged linear trend forecast.

    Clearly labelled a ``forecast`` (\"if the current trajectory continues\"),
    never a hard prediction. Reports INSUFFICIENT_DATA on thin history — no
    fabricated certainty.
    """
    arr = _to_numeric(values)
    n = arr.size
    if n < min_history:
        return {
            "availability": "INSUFFICIENT_DATA",
            "points": [],
            "note": f"Only {n} observations; need at least {min_history} to forecast.",
        }
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, arr, 1)
    points = []
    for i in range(1, horizon + 1):
        value = round(float(slope * (n + i) + intercept), 4)
        points.append({"timestamp": str(i), "value": value, "kind": "forecast"})
    return {
        "availability": "AVAILABLE",
        "points": points,
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "horizon": horizon,
        "note": "If the current trajectory continues (linear trend).",
    }