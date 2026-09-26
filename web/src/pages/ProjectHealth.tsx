import { useState } from "react";

type HealthEvidence = {
  metric: string;
  value?: number | null;
  previous_value?: number | null;
  delta?: number | null;
  timestamp?: string | null;
  window?: string | null;
  source?: string;
  confidence?: number;
  direction?: string;
  availability?: string;
};

type Dimension = {
  name: string;
  state: string;
  trend: string;
  confidence: number;
  score: number | null;
  availability: string;
  supporting_metrics?: string[];
  evidence?: HealthEvidence[];
};

type Risk = {
  name: string;
  level: string;
  trend: string;
  confidence: number;
  summary?: string;
};

type Signal = {
  name: string;
  dimension: string;
  severity?: string;
  description?: string;
};

type TimelineEvent = {
  timestamp?: string;
  event?: string;
  dimension?: string | null;
  metrics?: string[];
};

type ChangedMetric = {
  metric: string;
  dimension?: string | null;
  value?: number | null;
  previous_value?: number | null;
  delta?: number | null;
  relative_delta?: number | null;
  direction?: string;
  trend?: string;
  observation_window?: string | null;
  data_source?: string | null;
  availability?: string | null;
  confidence?: number | null;
};

type AnomalyPoint = {
  metric: string;
  timestamp?: string | null;
  observed?: number | null;
  baseline?: number | null;
  deviation?: number | null;
  description?: string;
  zscore?: number | null;
};

type ChangePoint = {
  metric: string;
  timestamp?: string | null;
  before_mean?: number | null;
  after_mean?: number | null;
  shift_magnitude?: number | null;
  description?: string;
};

type HistoryPoint = {
  timestamp: string;
  metric: string;
  value: number | null;
  observation_window?: string | null;
  availability?: string;
  confidence?: number;
};

type HealthEventRow = {
  kind: string;
  dimension?: string;
  severity?: string;
  summary?: string;
  observed_at?: string;
  evidence?: HealthEvidence[];
};

type CompareMetricRow = {
  metric: string;
  label: string;
  dimension: string;
  repository_a: number | null;
  repository_b: number | null;
  difference: number | null;
  interpretation: string;
  evidence?: {
    a?: { source?: string | null; window?: string | null; availability?: string };
    b?: { source?: string | null; window?: string | null; availability?: string };
  };
};

type CompareDimRow = {
  dimension: string;
  score_a: number | null;
  score_b: number | null;
  difference: number | null;
  state_a: string;
  state_b: string;
  interpretation: string;
};

type CompareResponse = {
  repository_a: string;
  repository_b: string;
  trajectory: { a: string | null; b: string | null; interpretation: string };
  overall_score: {
    a: number | null;
    b: number | null;
    state_a?: string;
    state_b?: string;
  };
  dimensions: CompareDimRow[];
  metrics: CompareMetricRow[];
};

type HistoryResponse = {
  health_key?: string;
  observations?: HistoryPoint[];
  by_metric?: Record<string, HistoryPoint[]>;
  count?: number;
  error?: string;
};

type HealthAnalysis = {
  analysis_id: string;
  owner: string;
  repository: string;
  full_name: string;
  url: string;
  created_at: string;
  overall_score: number | null;
  score_state?: string;
  trajectory: string;
  trajectory_confidence: number;
  health_events?: HealthEventRow[];
  dimensions: Dimension[];
  risks: Risk[];
  signals: Signal[];
  correlations?: { metric_a: string; metric_b: string; correlation_coefficient?: number }[];
  anomalies?: AnomalyPoint[];
  change_points?: ChangePoint[];
  trends?: { metric?: string; slope?: number; direction?: string; r2?: number }[];
  timeline: TimelineEvent[];
  why_changed?: ChangedMetric[];
  changes?: ChangedMetric[];
  contributors?: unknown[];
  releases?: unknown[];
  repository_metadata?: {
    owner?: string;
    name?: string;
    full_name?: string;
    url?: string;
    stars?: number | null;
    forks?: number | null;
    open_issues?: number | null;
    created_at?: string | null;
    pushed_at?: string | null;
    default_branch?: string | null;
    license?: string | null;
    archived?: boolean;
    description?: string | null;
  } | null;
  data_sufficiency?: {
    dimensions_computed?: number;
    dimensions_total?: number;
    coverage?: number;
    observation_window?: string;
    source_analysis?: string;
    source_details?: {
      dependencies?: string;
      security?: string;
      code_health?: string;
    };
  };
  ai_analysis?: string | null;
  ai_available?: boolean;
  gemini_analysis?: string | null;
  gemini_available?: boolean;
  reviewed?: boolean;
  review_notes?: string[];
};

function fmt(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

function arrow(trend: string): string {
  if (trend === "INCREASING" || trend === "IMPROVING") return "\u2191";
  if (trend === "DECREASING" || trend === "DETERIORATING") return "\u2193";
  if (trend === "VOLATILE") return "\u2195";
  if (trend === "STABLE") return "\u2192";
  return "\u00b7";
}

function stateLabel(s: string): string {
  switch (s) {
    case "GOOD":
      return "good";
    case "STABLE":
      return "stable";
    case "DETERIORATING":
      return "deteriorating";
    case "IMPROVING":
      return "improving";
    case "NOT_AVAILABLE":
      return "not available";
    default:
      return (s || "").toLowerCase();
  }
}
export function ProjectHealth() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<HealthAnalysis | null>(null);
  const [activeDim, setActiveDim] = useState<string | null>(null);

  async function runAnalysis() {
    if (!input.trim()) return;
    setLoading(true);
    setError(null);
    setAnalysis(null);
    setActiveDim(null);
    try {
      const res = await fetch("/api/project-health/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repository: input.trim(), include_source: true }),
      });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const j = await res.json();
          detail = j.detail || detail;
        } catch {
          /* keep default */
        }
        throw new Error(detail);
      }
      const data = (await res.json()) as HealthAnalysis;
      setAnalysis(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const dims = analysis?.dimensions ?? [];
  const active = activeDim ? (dims.find((d) => d.name === activeDim) ?? null) : null;

  return (
    <main className="ph-page">
      <p className="eyebrow">Project Health</p>
      <h1 className="ph-title">Analyze a GitHub Repository</h1>
      <p className="ph-sub">
        Enter a repository to get a multidimensional, evidence-first health
        analysis — activity, maintenance, community, releases, dependencies and
        security — built from real GitHub data, Git history and code intelligence.
      </p>

      <div className="ph-input-row">
        <input
          className="ph-input"
          placeholder="github.com/owner/repository  or  owner/repository"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && runAnalysis()}
        />
        <button
          className="ph-btn"
          onClick={runAnalysis}
          disabled={loading || !input.trim()}
        >
          {loading ? "Analyzing…" : "Analyze Health"}
        </button>
      </div>

      {error && (
        <div className="ph-error">
          <strong>Analysis failed:</strong> {error}
          <div className="ph-error-hint">
            If the repository is real, this is likely a GitHub rate limit, a
            network issue, or the repository has too little public history.
          </div>
        </div>
      )}

      <CompareView />

      {loading && (
        <div className="ph-loading">
          <div className="ph-loading-steps">
            <span>Data collection</span>
            <span>Feature engineering</span>
            <span>Temporal analysis</span>
            <span>Health intelligence</span>
          </div>
        </div>
      )}

      {analysis && !loading && (
        <Dashboard
          analysis={analysis}
          activeDim={activeDim}
          setActiveDim={setActiveDim}
          active={active}
        />
      )}
    </main>
  );
}
function Dashboard({
  analysis,
  activeDim,
  setActiveDim,
  active,
}: {
  analysis: HealthAnalysis;
  activeDim: string | null;
  setActiveDim: (s: string | null) => void;
  active: Dimension | null;
}) {
  const meta = analysis.repository_metadata;
  return (
    <div className="ph-dash">
      <section className="ph-card ph-hero-card">
        <div className="ph-repo-header">
          <h2>{analysis.full_name || `${analysis.owner}/${analysis.repository}`}</h2>
          {meta?.archived && <span className="ph-pill ph-pill-archived">archived</span>}
        </div>
        {meta?.description && <p className="ph-repo-desc">{meta.description}</p>}
        <div className="ph-score-row">
          <div className="ph-score">
            <div className="ph-score-num">
              {analysis.overall_score === null ? "—" : analysis.overall_score}
            </div>
            <div className="ph-score-label">Overall Health /100</div>
          </div>
          <div className="ph-score-meta">
            <MetaBits analysis={analysis} />
          </div>
        </div>
      </section>

      <section className="ph-card">
        <h3>Health Dimensions</h3>
        <div className="ph-dim-grid">
          {analysis.dimensions.map((d) => (
            <button
              key={d.name}
              className="ph-dim"
              onClick={() => setActiveDim(activeDim === d.name ? null : d.name)}
            >
              <div className="ph-dim-head">
                <span className="ph-dim-name">{d.name}</span>
                <span className="ph-dim-arrow">{arrow(d.trend)}</span>
              </div>
              <div className="ph-dim-score">
                {d.score === null ? "—" : fmt(d.score)}
              </div>
              <div className="ph-dim-state">{stateLabel(d.state)}</div>
              <div className="ph-dim-conf">confidence {fmt(d.confidence * 100, 0)}%</div>
            </button>
          ))}
        </div>
      </section>

      {active && (
        <section className="ph-card ph-drill">
          <div className="ph-drill-head">
            <h3>{active.name}</h3>
            <button className="ph-btn-ghost" onClick={() => setActiveDim(null)}>
              close
            </button>
          </div>
          <div className="ph-drill-meta">
            <span>state <b>{stateLabel(active.state)}</b></span>
            <span>trend <b>{active.trend}</b></span>
            <span>score <b>{active.score === null ? "—" : fmt(active.score)}</b></span>
            <span>availability <b>{active.availability}</b></span>
          </div>
          {active.supporting_metrics && active.supporting_metrics.length > 0 && (
            <div className="ph-drill-metrics">
              <b>Supporting metrics:</b> {active.supporting_metrics.join(", ")}
            </div>
          )}
          {active.evidence && active.evidence.length > 0 && (
            <table className="ph-ev-table">
              <thead>
                <tr>
                  <th>metric</th>
                  <th>value</th>
                  <th>prev</th>
                  <th>delta</th>
                  <th>source</th>
                </tr>
              </thead>
              <tbody>
                {active.evidence.slice(0, 12).map((e, i) => (
                  <tr key={i}>
                    <td>{e.metric}</td>
                    <td>{fmt(e.value)}</td>
                    <td>{fmt(e.previous_value)}</td>
                    <td>{fmt(e.delta)}</td>
                    <td>{e.source || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      )}

      <div className="ph-two-col">
        <section className="ph-card">
          <h3>Key Signals</h3>
          {analysis.signals.length === 0 ? (
            <p className="ph-empty">No significant signals detected.</p>
          ) : (
            <ul className="ph-signal-list">
              {analysis.signals.map((s, i) => (
                <li key={i} className={`ph-signal ${s.severity || "info"}`}>
                  <b>{s.name}</b>
                  {s.description && <span> — {s.description}</span>}
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="ph-card">
          <h3>Risk Areas</h3>
          <div className="ph-risk-list">
            {analysis.risks.map((r, i) => (
              <div key={i} className="ph-risk">
                <span className="ph-risk-name">{r.name}</span>
                <span className={`ph-badge ph-badge-${r.level.toLowerCase()}`}>{r.level}</span>
                <span className="ph-risk-trend">{arrow(r.trend)}</span>
                {r.summary && <span className="ph-risk-sum">{r.summary}</span>}
            </div>
          ))}
        </div>
      </section>
      </div>

      <section className="ph-card">
        <h3>Why Did Health Change?</h3>
        {analysis.why_changed && analysis.why_changed.length > 0 ? (
          <ul className="ph-why-list">
            {analysis.why_changed.map((c, i) => (
              <li key={i} className="ph-why-item">
                <span className="ph-why-rank">{i + 1}</span>
                <span className="ph-why-metric">
                  {c.metric}
                  {c.dimension && (
                    <span className="ph-why-dim">({c.dimension})</span>
                  )}
                </span>
                <span className="ph-why-arrow">
                  {arrow(c.direction || "")}
                </span>
                <span className="ph-why-values">
                  {typeof c.value === "number" ? fmt(c.value) : "—"}
                  {" → "}
                  {typeof c.previous_value === "number"
                    ? fmt(c.previous_value)
                    : "—"}
                </span>
                {c.relative_delta !== undefined && c.relative_delta !== null && (
                  <span className="ph-why-delta">
                    ({fmt(c.relative_delta * 100, 0)}%)
                  </span>
                )}
                <WhyExpand c={c} />
              </li>
            ))}
          </ul>
        ) : (
          <p className="ph-empty">
            No significant changes detected in the recent window.
          </p>
        )}
      </section>

      <SourceHealthSection analysis={analysis} />

      <HistoricalTrends analysis={analysis} />

      <AnomalyDetails analysis={analysis} />

      <section className="ph-card">
        <h3>Project Timeline</h3>
        <Timeline analysis={analysis} />
      </section>

      <section className="ph-card">
        <h3>AI Analysis</h3>
        <Interpretation analysis={analysis} />
      </section>

      <section className="ph-card">
        <h3>Data Sufficiency</h3>
        <div className="ph-suff">
          <Suff
            label="dimensions computed"
            value={`${analysis.data_sufficiency?.dimensions_computed ?? "—"}/${analysis.data_sufficiency?.dimensions_total ?? "—"}`}
          />
          <Suff
            label="coverage"
            value={analysis.data_sufficiency?.coverage !== undefined ? `${fmt(analysis.data_sufficiency.coverage * 100, 0)}%` : "—"}
          />
          <Suff
            label="observation window"
            value={analysis.data_sufficiency?.observation_window ?? "—"}
          />
        </div>
      </section>

      <section className="ph-card">
        <h3>Health Events</h3>
        <p className="ph-card-hint">
          Evidence-backed changes derived deterministically from the previous
          snapshot during monitoring refreshes. Statistical, not causal.
        </p>
        {analysis.health_events && analysis.health_events.length > 0 ? (
          <ul className="ph-events">
            {analysis.health_events.map((ev, i) => (
              <li
                key={`${ev.kind}-${ev.dimension}-${i}`}
                className={`ph-event ph-event-${ev.severity === "warning" ? "warn" : "info"}`}
              >
                <div className="ph-event-head">
                  <span className="ph-event-kind">{ev.kind}</span>
                  {ev.dimension && (
                    <span className="ph-event-dim">{ev.dimension}</span>
                  )}
                  {ev.observed_at && (
                    <span className="ph-event-time">
                      {ev.observed_at.slice(0, 10)}
                    </span>
                  )}
                </div>
                {ev.summary && <p className="ph-event-summary">{ev.summary}</p>}
                {ev.evidence && ev.evidence.length > 0 && (
                  <ul className="ph-event-evidence">
                    {ev.evidence.map((e, j) => (
                      <li key={j}>
                        <code>{e.metric}</code>
                        {e.previous_value !== null &&
                          e.previous_value !== undefined &&
                          " — "}
                        {e.previous_value !== null &&
                          e.previous_value !== undefined &&
                          `${fmt(e.previous_value)} → ${fmt(e.value)}`}
                        {e.source && (
                          <span className="ph-ev-src"> ({e.source})</span>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p className="ph-empty">
            No health events yet. Events appear after a monitoring refresh
            detects a statistically meaningful change versus the prior
            snapshot.
          </p>
        )}
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Analytical visualization (stage 2): real historical observations rendered
// as SVG time-series charts. No decorative charts — only real persisted data.
// When fewer than 3 observations exist for a metric, we say so explicitly.
// ---------------------------------------------------------------------------

function Sparkline({
  points,
  width = 220,
  height = 56,
  markAnomalies = [],
}: {
  points: HistoryPoint[];
  width?: number;
  height?: number;
  markAnomalies?: AnomalyPoint[];
}) {
  const values = points
    .map((p) => p.value)
    .filter((v): v is number => v !== null && v !== undefined);
  if (values.length < 2) {
    return (
      <div className="ph-chart-empty">
        <span>Insufficient historical data</span>
      </div>
    );
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pad = 4;
  const stepX = (width - 2 * pad) / (points.length - 1 || 1);
  const x = (i: number) => pad + i * stepX;
  const y = (v: number) => height - pad - ((v - min) / span) * (height - 2 * pad);
  const line = points
    .map((p, i) =>
      p.value === null || p.value === undefined ? null : `${x(i)},${y(p.value)}`
    )
    .filter(Boolean)
    .join(" ");
  const anomalySet = new Set(
    markAnomalies
      .map((a) => (a.timestamp ? new Date(a.timestamp).toISOString().slice(0, 10) : null))
      .filter(Boolean) as string[]
  );
  return (
    <svg
      className="ph-spark"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
    >
      {points.map((p, i) => {
        const day = p.timestamp ? new Date(p.timestamp).toISOString().slice(0, 10) : "";
        const isAnom = anomalySet.has(day);
        return (
          <circle
            key={i}
            cx={x(i)}
            cy={y(p.value ?? min)}
            r={isAnom ? 3.5 : 1.8}
            fill={isAnom ? "var(--alert)" : "var(--ink-40)"}
          />
        );
      })}
      <polyline points={line} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
    </svg>
  );
}

function HistoryChart({
  title,
  series,
  anomalies,
  changePoints,
}: {
  title: string;
  series: HistoryPoint[];
  anomalies: AnomalyPoint[];
  changePoints: ChangePoint[];
}) {
  const values = series
    .map((p) => p.value)
    .filter((v): v is number => v !== null && v !== undefined);
  const last = series[series.length - 1]?.value;
  return (
    <div className="ph-chart">
      <div className="ph-chart-head">
        <span className="ph-chart-title">{title}</span>
        <span className="ph-chart-last">
          last {last === null || last === undefined ? "—" : fmt(last)}
        </span>
      </div>
      <Sparkline points={series} markAnomalies={anomalies} />
      {(anomalies.length > 0 || changePoints.length > 0) && (
        <div className="ph-chart-events">
          {changePoints.length > 0 && (
            <span className="ph-chart-cp" title={changePoints[0].description || "Regime change"}>
              ⚡ change point
            </span>
          )}
          {anomalies.length > 0 && (
            <span className="ph-chart-anom" title={anomalies[0].description || "Anomaly"}>
              ● {anomalies.length} anomalous
            </span>
          )}
        </div>
      )}
      {values.length < 3 && (
        <div className="ph-chart-note">Fewer than 3 observations — trend not yet reliable.</div>
      )}
    </div>
  );
}

/** Expandable drill-down for one why-changed metric: value, previous, delta,
 *  relative delta, direction, observation window, data source, confidence. */
function WhyExpand({ c }: { c: Record<string, unknown> }) {
  const [open, setOpen] = useState(false);
  const num = (k: string) =>
    typeof c[k] === "number" ? fmt(c[k] as number, 3) : "\u2014";
  const rel = typeof c.relative_delta === "number"
    ? `${fmt(c.relative_delta * 100, 0)}%`
    : "\u2014";
  return (
    <span className="ph-why-expand">
      <button
        type="button"
        className="ph-why-toggle"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        {open ? "\u25B4 details" : "\u25BE details"}
      </button>
      {open && (
        <>
          <dl className="ph-why-grid">
            <dt>current</dt><dd>{num("value")}</dd>
            <dt>previous</dt><dd>{num("previous_value")}</dd>
            <dt>delta</dt><dd>{num("delta")}</dd>
            <dt>relative</dt><dd>{rel}</dd>
            <dt>trend</dt>
            <dd>{typeof c.direction === "string" ? c.direction : "\u2014"}</dd>
            <dt>window</dt><dd>{typeof c.window === "string" ? c.window : "\u2014"}</dd>
            <dt>source</dt><dd>{typeof c.source === "string" ? c.source : "\u2014"}</dd>
            <dt>confidence</dt>
            <dd>{typeof c.confidence === "number" ? fmt(c.confidence, 2) : "\u2014"}</dd>
          </dl>
          <a
            className="ph-why-link"
            href="#trends"
            onClick={(e) => {
              e.preventDefault();
              document.getElementById("trends")?.scrollIntoView({ behavior: "smooth" });
            }}
          >
            view history &rarr;
          </a>
        </>
      )}
    </span>
  );
}

type WindowKey = "30d" | "90d" | "6mo" | "1yr";

const WINDOWS: { key: WindowKey; days: number; label: string }[] = [
  { key: "30d", days: 30, label: "30 days" },
  { key: "90d", days: 90, label: "90 days" },
  { key: "6mo", days: 180, label: "6 months" },
  { key: "1yr", days: 365, label: "1 year" },
];

function HistoricalTrends({ analysis }: { analysis: HealthAnalysis }) {
  const [winKey, setWinKey] = useState<WindowKey>("90d");
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const ownerRepo = `${analysis.owner}/${analysis.repository}`;

  async function loadHistory(days: number) {
    const cutoff = new Date(Date.now() - days * 86400_000).toISOString();
    setLoading(true);
    setHistoryError(null);
    try {
      const res = await fetch(
        `/api/project-health/${encodeURIComponent(ownerRepo)}/history?limit=500`
      );
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const j = (await res.json()) as HistoryResponse;
      if (j && Array.isArray(j.observations)) {
        j.observations = j.observations.filter(
          (o) => !o.timestamp || o.timestamp >= cutoff
        );
      }
      setHistory(j);
    } catch (e) {
      setHistoryError(e instanceof Error ? e.message : String(e));
      setHistory(null);
    } finally {
      setLoading(false);
    }
  }

  const win = WINDOWS.find((w) => w.key === winKey) ?? WINDOWS[1];
  const observations = history?.observations ?? [];
  const byMetric: Record<string, HistoryPoint[]> = {};
  for (const obs of observations) {
    if (obs.value === null || obs.value === undefined) continue;
    (byMetric[obs.metric] ??= []).push(obs);
  }
  for (const k of Object.keys(byMetric)) {
    byMetric[k].sort((a, b) => (a.timestamp < b.timestamp ? -1 : 1));
  }

  const metricMeta: { key: string; title: string; anomalies: AnomalyPoint[]; cps: ChangePoint[] }[] = [
    {
      key: "rolling_30d",
      title: "Activity (30d commit rolling)",
      anomalies: (analysis.anomalies ?? []).filter((a) => a.metric === "commits"),
      cps: (analysis.change_points ?? []).filter((c) => c.metric === "commits"),
    },
    {
      key: "hhi",
      title: "Contributor concentration (HHI)",
      anomalies: (analysis.anomalies ?? []).filter((a) => a.metric === "hhi"),
      cps: (analysis.change_points ?? []).filter((c) => c.metric === "hhi"),
    },
    {
      key: "median_merge_days",
      title: "PR responsiveness (median merge days)",
      anomalies: (analysis.anomalies ?? []).filter((a) => a.metric === "median_merge_days"),
      cps: (analysis.change_points ?? []).filter((c) => c.metric === "median_merge_days"),
    },
    {
      key: "median_interval_days",
      title: "Release interval (median days)",
      anomalies: (analysis.anomalies ?? []).filter((a) => a.metric === "releases"),
      cps: (analysis.change_points ?? []).filter((c) => c.metric === "releases"),
    },
    {
      key: "median_resolution_days",
      title: "Issue resolution time (median days)",
      anomalies: (analysis.anomalies ?? []).filter((a) => a.metric === "issues"),
      cps: (analysis.change_points ?? []).filter((c) => c.metric === "issues"),
    },
    {
      key: "total_dependencies",
      title: "Dependency risk (dependency count)",
      anomalies: [],
      cps: [],
    },
    {
      key: "overall_score",
      title: "Overall health score",
      anomalies: [],
      cps: [],
    },
    {
      key: "findings_count",
      title: "Security trend (findings)",
      anomalies: (analysis.anomalies ?? []).filter((a) => a.metric === "findings_count"),
      cps: (analysis.change_points ?? []).filter((c) => c.metric === "findings_count"),
    },
    {
      key: "files_parsed",
      title: "Code health (source files understood)",
      anomalies: [],
      cps: [],
    },
  ];

  return (
    <section className="ph-card" id="trends">
      <div className="ph-chart-section-head">
        <h3>Health Trajectory</h3>
        <div className="ph-win-tabs">
          {WINDOWS.map((w) => (
            <button
              key={w.key}
              className={`ph-win-tab ${w.key === winKey ? "active" : ""}`}
              onClick={() => {
                setWinKey(w.key);
                loadHistory(w.days);
              }}
            >
              {w.label}
            </button>
          ))}
        </div>
      </div>
      <p className="ph-chart-intro">
        These series reflect real, persisted observations across analysis runs.
        A single run produces one observation per metric, so trend charts
        appear only after multiple runs accumulate over time.
      </p>

      {loading && <div className="ph-loading-steps"><span>Loading history…</span></div>}

      {historyError ? (
        <p className="ph-empty">Historical observations not yet available ({historyError}).</p>
      ) : history === null ? (
        <div className="ph-chart-cta">
          <button className="ph-btn-ghost" onClick={() => loadHistory(win.days)}>
            Load historical trend charts
          </button>
          <span className="ph-chart-cta-hint">
            Uses the {win.label} observation window.
          </span>
        </div>
      ) : (
        <div className="ph-chart-grid">
          {metricMeta.map((m) => {
            const series = byMetric[m.key] ?? [];
            if (series.length === 0) {
              return (
                <div key={m.key} className="ph-chart">
                  <div className="ph-chart-head">
                    <span className="ph-chart-title">{m.title}</span>
                  </div>
                  <div className="ph-chart-empty">
                    <span>Insufficient historical data</span>
                  </div>
                </div>
              );
            }
            return (
              <HistoryChart
                key={m.key}
                title={m.title}
                series={series}
                anomalies={m.anomalies}
                changePoints={m.cps}
              />
            );
          })}
        </div>
      )}
    </section>
  );
}

function SourceHealthSection({ analysis }: { analysis: HealthAnalysis }) {
  const sd = analysis.data_sufficiency?.source_details;
  const src = analysis.data_sufficiency?.source_analysis;
  const badge = (v?: string) =>
    !v
      ? "—"
      : v === "AVAILABLE" || v === "available" || v === "SOURCE_ANALYSIS_AVAILABLE"
        ? "✓"
        : "·";
  const label = (v?: string) =>
    !v
      ? "not requested"
      : v === "SOURCE_ANALYSIS_AVAILABLE" || v === "SOURCE_ANALYSIS_UNAVAILABLE"
        ? v.replace("SOURCE_ANALYSIS_", "").toLowerCase()
        : v.toLowerCase();
  return (
    <section className="ph-card">
      <h3>Source Intelligence</h3>
      {!src ? (
        <p className="ph-empty">
          Source analysis was not performed. This analysis covers GitHub
          activity metadata only. Rerun with source analysis enabled to include
          dependency, security and code-health signals.
        </p>
      ) : (
        <>
          <div className="ph-src-status">
            <span
              className={`ph-src-badge ${src === "SOURCE_ANALYSIS_AVAILABLE" ? "ok" : "na"}`}
            >
              {src === "SOURCE_ANALYSIS_AVAILABLE"
                ? "SOURCE_ANALYSIS_AVAILABLE"
                : "SOURCE_ANALYSIS_UNAVAILABLE"}
            </span>
          </div>
          <div className="ph-src-grid">
            <div className="ph-src-item">
              <span className="ph-src-label">Dependencies</span>
              <span className={`ph-src-val ${sd?.dependencies === "AVAILABLE" ? "ok" : ""}`}>
                {badge(sd?.dependencies)} {label(sd?.dependencies)}
              </span>
            </div>
            <div className="ph-src-item">
              <span className="ph-src-label">Security</span>
              <span
                className={`ph-src-val ${sd?.security === "available" || sd?.security === "no_findings" ? "ok" : ""}`}
              >
                {badge(sd?.security)} {label(sd?.security)}
              </span>
            </div>
            <div className="ph-src-item">
              <span className="ph-src-label">Code health</span>
              <span className={`ph-src-val ${sd?.code_health === "AVAILABLE" ? "ok" : ""}`}>
                {badge(sd?.code_health)} {label(sd?.code_health)}
              </span>
            </div>
          </div>
        </>
      )}
    </section>
  );
}

function AnomalyDetails({ analysis }: { analysis: HealthAnalysis }) {
  const anomalies = analysis.anomalies ?? [];
  const changePoints = analysis.change_points ?? [];
  if (anomalies.length === 0 && changePoints.length === 0) {
    return (
      <section className="ph-card">
        <h3>Anomalies & Regime Changes</h3>
        <p className="ph-empty">No anomalies or change points detected in the current data.</p>
      </section>
    );
  }
  return (
    <section className="ph-card">
      <h3>Anomalies & Regime Changes</h3>
      {anomalies.length > 0 && (
        <div className="ph-anom-list">
          {anomalies.slice(0, 10).map((a, i) => (
            <details key={i} className="ph-anom-item">
              <summary>
                <span className="ph-anom-metric">{a.metric}</span>
                {a.timestamp && (
                  <span className="ph-anom-date">{new Date(a.timestamp).toLocaleDateString()}</span>
                )}
                <span className="ph-anom-flag">anomaly</span>
              </summary>
              <div className="ph-anom-body">
                {a.observed !== null && a.observed !== undefined && (
                  <div>observed: <b>{fmt(a.observed, 2)}</b></div>
                )}
                {a.baseline !== null && a.baseline !== undefined && (
                  <div>baseline: <b>{fmt(a.baseline, 2)}</b></div>
                )}
                {a.deviation !== null && a.deviation !== undefined && (
                  <div>deviation: <b>{fmt(a.deviation, 2)}</b></div>
                )}
                {a.zscore !== null && a.zscore !== undefined && (
                  <div>z-score: <b>{fmt(a.zscore, 2)}</b></div>
                )}
                {a.description && <div className="ph-anom-desc">{a.description}</div>}
              </div>
            </details>
          ))}
        </div>
      )}
      {changePoints.length > 0 && (
        <div className="ph-cp-list">
          {changePoints.slice(0, 6).map((cp, i) => (
            <details key={i} className="ph-cp-item">
              <summary>
                <span className="ph-cp-metric">{cp.metric}</span>
                {cp.timestamp && (
                  <span className="ph-anom-date">{new Date(cp.timestamp).toLocaleDateString()}</span>
                )}
                <span className="ph-cp-flag">regime change</span>
              </summary>
              <div className="ph-anom-body">
                {cp.before_mean !== null && cp.before_mean !== undefined && (
                  <div>before: <b>{fmt(cp.before_mean, 2)}</b></div>
                )}
                {cp.after_mean !== null && cp.after_mean !== undefined && (
                  <div>after: <b>{fmt(cp.after_mean, 2)}</b></div>
                )}
                {cp.shift_magnitude !== null && cp.shift_magnitude !== undefined && (
                  <div>shift: <b>{fmt(cp.shift_magnitude, 2)}</b></div>
                )}
                {cp.description && <div className="ph-anom-desc">{cp.description}</div>}
              </div>
            </details>
          ))}
        </div>
      )}
      <p className="ph-chart-note">
        Anomalies and change points are detected by the temporal engine using
        robust statistics (MAD-based z-scores and mean-shift tests). These are
        statistically significant behavior signals — not claims of causation.
      </p>
    </section>
  );
}

function Suff({ label, value }: { label: string; value: string }) {
  return (
    <div className="ph-suff-item">
      <span className="ph-suff-value">{value}</span>
      <span className="ph-suff-label">{label}</span>
    </div>
  );
}

function MetaBits({ analysis }: { analysis: HealthAnalysis }) {
  const meta = analysis.repository_metadata;
  return (
    <div className="ph-meta-bits">
      <div><b>Trajectory:</b> {analysis.trajectory} {arrow(analysis.trajectory)}</div>
      <div><b>Confidence:</b> {fmt(analysis.trajectory_confidence * 100, 0)}%</div>
      {meta && (
        <>
          <div><b>Stars:</b> {meta.stars ?? "—"}</div>
          <div><b>Forks:</b> {meta.forks ?? "—"}</div>
          <div><b>Open issues:</b> {meta.open_issues ?? "—"}</div>
          <div><b>Branch:</b> {meta.default_branch ?? "—"}</div>
          <div><b>License:</b> {meta.license ?? "—"}</div>
          {meta.created_at && <div><b>Created:</b> {new Date(meta.created_at).toLocaleDateString()}</div>}
        </>
      )}
      <div><b>Analysis ID:</b> {analysis.analysis_id.slice(0, 8)}</div>
    </div>
  );
}

function Timeline({ analysis }: { analysis: HealthAnalysis }) {
  const events = analysis.timeline;
  if (events.length === 0) return <p className="ph-empty">No timeline events derived.</p>;
  return (
    <ul className="ph-timeline">
      {events.slice(0, 12).map((ev, i) => (
        <li key={i} className="ph-timeline-item">
          <span className="ph-timeline-dim">{ev.dimension || "health"}</span>
          <span className="ph-timeline-event">{ev.event}</span>
          {ev.metrics && ev.metrics.length > 0 && (
            <span className="ph-timeline-metrics">{ev.metrics.join(", ")}</span>
          )}
        </li>
      ))}
    </ul>
  );
}

function CompareView() {
  const [repoA, setRepoA] = useState("");
  const [repoB, setRepoB] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResponse | null>(null);

  async function runCompare() {
    if (!repoA.trim() || !repoB.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/project-health/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repository_a: repoA.trim(),
          repository_b: repoB.trim(),
        }),
      });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const j = await res.json();
          detail = j.detail || detail;
        } catch {
          /* keep default */
        }
        throw new Error(detail);
      }
      setResult((await res.json()) as CompareResponse);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const fmtCell = (v: number | null | undefined) =>
    v === null || v === undefined ? "—" : fmt(v, Math.abs(v) < 10 ? 2 : 0);

  return (
    <section className="ph-card ph-compare">
      <h3>Compare Projects</h3>
      <p className="ph-compare-hint">
        Deterministic comparison of the two most recent health snapshots, using
        the same normalized metrics that drive the health score. Missing data is
        reported explicitly — never treated as zero.
      </p>
      <div className="ph-input-row">
        <input
          className="ph-input"
          placeholder="Repository A — github.com/owner/repository"
          value={repoA}
          onChange={(e) => setRepoA(e.target.value)}
        />
        <input
          className="ph-input"
          placeholder="Repository B — github.com/owner/repository"
          value={repoB}
          onChange={(e) => setRepoB(e.target.value)}
        />
        <button
          className="ph-btn"
          onClick={runCompare}
          disabled={loading || !repoA.trim() || !repoB.trim()}
        >
          {loading ? "Comparing…" : "Compare"}
        </button>
      </div>

      {error && <div className="ph-error">{error}</div>}

      {result && (
        <>
          <div className="ph-compare-head">
            <div className="ph-compare-side">
              <span className="ph-compare-tag">A</span>
              <b>{result.repository_a}</b>
              <span className="ph-score-num">{fmt(result.overall_score.a)}</span>
            </div>
            <div className="ph-compare-side">
              <span className="ph-compare-tag">B</span>
              <b>{result.repository_b}</b>
              <span className="ph-score-num">{fmt(result.overall_score.b)}</span>
            </div>
          </div>
          <p className="ph-compare-traj">{result.trajectory.interpretation}</p>

          <table className="ph-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>A</th>
                <th>B</th>
                <th>Interpretation</th>
                <th>Evidence</th>
              </tr>
            </thead>
            <tbody>
              {result.dimensions.map((d) => (
                <tr key={`dim-${d.dimension}`} className="ph-compare-dimrow">
                  <td>
                    <b>{d.dimension}</b> <span className="ph-dim-note">(score)</span>
                  </td>
                  <td>{fmtCell(d.score_a)}</td>
                  <td>{fmtCell(d.score_b)}</td>
                  <td>{d.interpretation}</td>
                  <td className="ph-evid-note">
                    {d.state_a} vs {d.state_b}
                  </td>
                </tr>
              ))}
              {result.metrics.map((m) => (
                <tr key={`met-${m.metric}`}>
                  <td>
                    {m.label}
                    <span className="ph-dim-note"> · {m.dimension}</span>
                  </td>
                  <td>{fmtCell(m.repository_a)}</td>
                  <td>{fmtCell(m.repository_b)}</td>
                  <td>{m.interpretation}</td>
                  <td className="ph-evid-note">
                    {[
                      m.evidence?.a?.source,
                      m.evidence?.a?.window,
                      m.evidence?.a?.availability,
                    ]
                      .filter(Boolean)
                      .join(" · ") || "not available"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}

function Interpretation({ analysis }: { analysis: HealthAnalysis }) {
  const text = analysis.gemini_analysis || analysis.ai_analysis;
  if (text) {
    return (
      <div className="ph-ai-block">
        <div className="ph-ai-tag">
          {analysis.gemini_available ? "Gemini interpretation" : "AI interpretation"}
        </div>
        <p className="ph-ai-text">{text}</p>
        {analysis.reviewed && <div className="ph-reviewed">Reviewed ✓</div>}
        {analysis.review_notes && analysis.review_notes.length > 0 && (
          <ul className="ph-review-notes">
            {analysis.review_notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        )}
      </div>
    );
  }
  return (
    <div className="ph-ai-fallback">
      <p>
        <b>AI interpretation not available.</b> This happens when Gemini is not
        configured or is unavailable. The deterministic health dimensions and
        evidence above remain valid — they are computed from real data and never
        depend on the AI.
      </p>
      <p className="ph-ai-fallback-note">
        RepoIntel treats the AI strictly as an interpreter. It never invents
        scores, counts, trends or repository facts; those are produced by the
        statistical engine from real observations.
      </p>
    </div>
  );
}