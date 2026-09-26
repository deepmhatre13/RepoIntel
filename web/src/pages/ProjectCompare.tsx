import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

type MetricRow = {
  metric: string;
  label: string;
  dimension: string;
  repository_a: number | null;
  repository_b: number | null;
  difference: number | null;
  interpretation: string;
  evidence: {
    a: { source: string | null; window: string | null; availability: string };
    b: { source: string | null; window: string | null; availability: string };
  };
};

type CompareResult = {
  repository_a?: string;
  repository_b?: string;
  overall_score: { a: number | null; b: number | null; state_a: string; state_b: string };
  trajectory: { a: string | null; b: string | null; interpretation: string };
  dimensions: Array<{
    dimension: string;
    score_a: number | null;
    score_b: number | null;
    difference: number | null;
    state_a: string;
    state_b: string;
    interpretation: string;
  }>;
  metrics: MetricRow[];
};

const fmt = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : `${Math.round(v * 100) / 100}`;

const AVAIL_OK = (a: string) => a === "AVAILABLE" || a === "available";

export function ProjectCompare() {
  const [params, setParams] = useSearchParams();
  const a = params.get("a") ?? "";
  const b = params.get("b") ?? "";
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runCompare = (e?: React.FormEvent) => {
    e?.preventDefault();
    const va = (document.getElementById("cmp-a") as HTMLInputElement)?.value;
    const vb = (document.getElementById("cmp-b") as HTMLInputElement)?.value;
    if (!va || !vb || va === vb) {
      setError("Please provide two different GitHub repositories.");
      return;
    }
    setError(null);
    setResult(null);
    setParams({ a: va, b: vb });
  };

  const doCompare = async () => {
    if (!a || !b || a === b) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/project-health/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repository_a: a, repository_b: b }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => null);
        throw new Error(d?.detail || `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void doCompare(); }, [a, b]);

  const canCompare = !!(a && b && a !== b);

  return (
    <div className="ph-wrap">
      <section className="ph-card ph-hero-card">
        <h2>Compare Projects</h2>
        <p className="ph-card-hint">
          Compare two repositories' Project Health using normalized,
          rate/level-based metrics. Missing data is reported explicitly
          rather than treated as zero.
        </p>
        <form
          className="ph-input-row"
          onSubmit={runCompare}
          style={{ gridTemplateColumns: "1fr 1fr auto" }}
        >
          <input
            id="cmp-a"
            className="ph-input"
            placeholder="owner/repository or GitHub URL (A)"
            defaultValue={a}
          />
          <input
            id="cmp-b"
            className="ph-input"
            placeholder="owner/repository or GitHub URL (B)"
            defaultValue={b}
          />
          <button
            type="submit"
            className="ph-btn"
            disabled={loading}
            style={{ whiteSpace: "nowrap" }}
          >
            {loading ? "Comparing…" : "Compare"}
          </button>
        </form>
        {error && <p className="ph-empty">{error}</p>}
        {!canCompare && !loading && !error && (
          <p className="ph-empty">
            Enter two different repositories above to begin.
          </p>
        )}
            </section>

      {result && (
        <>
          <section className="ph-card">
            <h3>Overall Score</h3>
            <div className="ph-score-comparison">
              <div className="ph-comp-cell">
                <span className="ph-comp-repo">{result.repository_a}</span>
                <span className="ph-comp-score">
                  {fmt(result.overall_score?.a)} ({result.overall_score?.state_a})
                </span>
              </div>
              <div className="ph-comp-vs">vs</div>
              <div className="ph-comp-cell">
                <span className="ph-comp-repo">{result.repository_b}</span>
                <span className="ph-comp-score">
                  {fmt(result.overall_score?.b)} ({result.overall_score?.state_b})
                </span>
              </div>
            </div>
          </section>

          <section className="ph-card">
            <h3>Trajectory</h3>
            <p>{result.trajectory.interpretation}</p>
            <div className="ph-comp-metrics">
              <span>A: {result.trajectory.a ?? "—"}</span>
              <span>B: {result.trajectory.b ?? "—"}</span>
            </div>
          </section>

          <section className="ph-card">
            <h3>Dimensions</h3>
            <div className="ph-comp-metrics">
              <div className="ph-comp-row ph-comp-row-head">
                <span>Metric</span>
                <span className="ph-comp-cell">A</span>
                <span className="ph-comp-cell">B</span>
                <span>Interpretation</span>
              </div>
              {result.dimensions.map((d) => (
                <div className="ph-comp-row" key={d.dimension}>
                  <span>{d.dimension}</span>
                  <span className="ph-comp-cell">
                    {fmt(d.score_a)} ({d.state_a})
                  </span>
                  <span className="ph-comp-cell">
                    {fmt(d.score_b)} ({d.state_b})
                  </span>
                  <span>{d.interpretation}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="ph-card">
            <h3>Metric Comparison</h3>
            <div className="ph-comp-metrics">
              <div className="ph-comp-row ph-comp-row-head">
                <span>Metric</span>
                <span className="ph-comp-cell">A</span>
                <span className="ph-comp-cell">B</span>
                <span>Interpretation</span>
              </div>
              {result.metrics.map((m) => (
                <div className="ph-comp-row" key={m.metric}>
                  <span title={m.dimension}>{m.label}</span>
                  <span className="ph-comp-cell">
                    {m.repository_a === null ? "unavailable" : fmt(m.repository_a)}
                  </span>
                  <span className="ph-comp-cell">
                    {m.repository_b === null ? "unavailable" : fmt(m.repository_b)}
                  </span>
                  <span>
                    <span className="ph-ev-src">
                      A:{" "}
                      {AVAIL_OK(m.evidence.a.availability)
                        ? m.evidence.a.source
                        : m.evidence.a.availability}
                      {" "}
                      B:{" "}
                      {AVAIL_OK(m.evidence.b.availability)
                        ? m.evidence.b.source
                        : m.evidence.b.availability}
                    </span>
                    <br />
                    {m.interpretation}
                  </span>
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      <p className="ph-empty" style={{ fontSize: "0.8em" }}>
        <Link to="/health">← Back to Project Health</Link>
      </p>
    </div>
  );
}

