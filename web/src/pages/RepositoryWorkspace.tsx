import { FormEvent, ReactNode, useEffect, useState } from "react";
import { NavLink, useParams, useSearchParams } from "react-router-dom";
import {
  askRepository,
  getArtifacts,
  getDependencies,
  getGraph,
  getRepositoryStatus,
  getRun,
  getSymbols,
  type RepositoryStatus,
  type RunDetail,
} from "../lib/api";

type Section = "overview" | "readme" | "architecture" | "dependencies" | "symbols" | "graph" | "chat" | "runs";

const sections: Array<{ id: Section; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "readme", label: "README" },
  { id: "architecture", label: "Architecture" },
  { id: "dependencies", label: "Dependencies" },
  { id: "symbols", label: "Symbols" },
  { id: "graph", label: "Graph" },
  { id: "chat", label: "Chat" },
  { id: "runs", label: "Agent Runs" },
];

function formatRepository(url: string) {
  return url.replace(/^https?:\/\/(www\.)?github\.com\//, "").replace(/\.git$/, "");
}

function EmptyState({ children }: { children: ReactNode }) {
  return <p className="workspace-empty">{children}</p>;
}

export function RepositoryWorkspace() {
  const { repositoryId, section = "overview" } = useParams<{ repositoryId: string; section?: string }>();
  const activeSection = (sections.some((item) => item.id === section) ? section : "overview") as Section;
  const [searchParams] = useSearchParams();
  const [status, setStatus] = useState<RepositoryStatus | null>(null);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [content, setContent] = useState<unknown>(null);
  const [error, setError] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<{ text: string; evidence?: unknown[] } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!repositoryId) return;
    let cancelled = false;
    setLoading(true);
    setError("");
    setContent(null);
    const load = async () => {
      try {
        const currentStatus = await getRepositoryStatus(repositoryId);
        if (cancelled) return;
        setStatus(currentStatus);
        const runId = searchParams.get("run") || currentStatus.last_run_id || currentStatus.recent_runs?.[0]?.run_id;
        if (runId) setRun(await getRun(runId));
        if (activeSection === "readme") setContent((await getArtifacts(repositoryId)).artifacts[0] || null);
        if (activeSection === "dependencies") setContent((await getDependencies(repositoryId)).dependencies);
        if (activeSection === "symbols") setContent((await getSymbols(repositoryId)).symbols);
        if (activeSection === "graph" || activeSection === "architecture") setContent((await getGraph(repositoryId)).neighbors);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Repository data is unavailable.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => { cancelled = true; };
  }, [repositoryId, activeSection, searchParams]);

  useEffect(() => {
    if (!repositoryId || !run || ["COMPLETE", "FAILED", "PARTIAL"].includes(run.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await getRun(run.run_id);
        setRun(next);
        if (next.status === "COMPLETE") setStatus(await getRepositoryStatus(repositoryId));
      } catch {
        // The next navigation or reload will surface a durable API error.
      }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [repositoryId, run]);

  async function submitQuestion(event: FormEvent) {
    event.preventDefault();
    if (!repositoryId || !question.trim()) return;
    setError("");
    try {
      const result = await askRepository(repositoryId, question.trim());
      setAnswer({ text: result.answer || result.response || "No answer was returned.", evidence: result.evidence });
      setQuestion("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat is unavailable.");
    }
  }

  if (!repositoryId) return <EmptyState>No repository selected.</EmptyState>;
  const repoName = status ? formatRepository(status.repo_url) : "Repository";

  return (
    <div className="workspace-shell">
      <aside className="workspace-sidebar">
        <div className="workspace-context"><p className="eyebrow">Repository</p><strong>{repoName}</strong><span>{status?.archetype || "Analysis pending"}</span></div>
        <nav aria-label="Repository workspace">
          {sections.map((item) => <NavLink key={item.id} to={`/app/repos/${repositoryId}/${item.id}${searchParams.get("run") ? `?run=${searchParams.get("run")}` : ""}`} className={({ isActive }) => isActive ? "workspace-nav active" : "workspace-nav"}>{item.label}</NavLink>)}
        </nav>
        <div className="workspace-advanced"><p className="eyebrow">Advanced Analysis</p><NavLink to={`/health?repository=${repositoryId}`}>Project Health</NavLink><NavLink to={`/health/compare?repository=${repositoryId}`}>Compare Projects</NavLink></div>
      </aside>
      <main className="workspace-main">
        <header className="workspace-header"><div><p className="eyebrow">{activeSection === "runs" ? "Agent execution" : "Repository workspace"}</p><h1>{activeSection === "overview" ? repoName : sections.find((item) => item.id === activeSection)?.label}</h1></div><span className={`status-mark status-${(status?.status || run?.status || "pending").toLowerCase()}`}>{status?.status || run?.status || "PENDING"}</span></header>
        {error ? <p className="form-error" role="alert">{error}</p> : null}
        {loading && !status ? <EmptyState>Loading observed repository data...</EmptyState> : null}
        {activeSection === "overview" ? <Overview status={status} run={run} /> : null}
        {activeSection === "readme" ? <Readme artifact={content as { content: string; version: number; created_at: string } | null} /> : null}
        {activeSection === "dependencies" ? <DataList title="Dependencies" items={content as Array<Record<string, unknown>> | null} fields={["package_name", "ecosystem", "version_spec", "dependency_type"]} /> : null}
        {activeSection === "symbols" ? <DataList title="Symbols" items={content as Array<Record<string, unknown>> | null} fields={["symbol_name", "symbol_type", "file_path", "line_start", "line_end"]} /> : null}
        {activeSection === "graph" || activeSection === "architecture" ? <GraphView items={content as Array<Record<string, unknown>> | null} architecture={activeSection === "architecture"} /> : null}
        {activeSection === "chat" ? <Chat question={question} setQuestion={setQuestion} submit={submitQuestion} answer={answer} /> : null}
        {activeSection === "runs" ? <Runs status={status} run={run} /> : null}
      </main>
    </div>
  );
}

function Overview({ status, run }: { status: RepositoryStatus | null; run: RunDetail | null }) {
  const counts = status?.artifact_counts || {};
  const metrics = [["Files", counts.files || counts.file_count], ["Symbols", counts.symbols || counts.symbol_count], ["Chunks", counts.chunks || counts.chunk_count], ["Dependencies", counts.dependencies || counts.dependency_count]];
  return <section className="workspace-content"><div className="metric-strip">{metrics.map(([label, value]) => <div className="metric" key={label}><strong>{value ?? "Not available"}</strong><span>{label}</span></div>)}</div><div className="workspace-columns"><article className="workspace-section"><p className="eyebrow">Repository facts</p><dl className="facts"><dt>URL</dt><dd>{status?.repo_url || "Not available"}</dd><dt>Archetype</dt><dd>{status?.archetype || "Insufficient evidence"}</dd><dt>Branch</dt><dd>{status?.branch || "Not available"}</dd><dt>Latest commit</dt><dd>{status?.head_commit || "Not available"}</dd></dl></article><article className="workspace-section"><p className="eyebrow">Latest analysis</p><p>{run?.current_stage || run?.status || "No analysis run found"}</p><p className="muted">Run state is read from the backend lifecycle. No synthetic progress is shown.</p></article></div></section>;
}

function Readme({ artifact }: { artifact: { content: string; version: number; created_at: string } | null }) {
  if (!artifact) return <EmptyState>No generated README is available yet. Complete an analysis run to create the primary artifact.</EmptyState>;
  return <section className="artifact-view"><div className="artifact-meta">Generated from repository evidence · version {artifact.version} · {new Date(artifact.created_at).toLocaleString()}</div><pre>{artifact.content}</pre></section>;
}

function DataList({ title, items, fields }: { title: string; items: Array<Record<string, unknown>> | null; fields: string[] }) {
  if (!items?.length) return <EmptyState>No {title.toLowerCase()} were discovered in the indexed repository.</EmptyState>;
  return <section className="data-table"><h2>{title}</h2>{items.map((item, index) => <div className="data-row" key={`${String(item[fields[0]])}-${index}`}>{fields.map((field) => <span key={field}><small>{field.replaceAll("_", " ")}</small>{String(item[field] ?? "Not available")}</span>)}</div>)}</section>;
}

function GraphView({ items, architecture }: { items: Array<Record<string, unknown>> | null; architecture: boolean }) {
  if (!items?.length) return <EmptyState>No graph relationships are available for this repository yet.</EmptyState>;
  return <section className="data-table"><p className="eyebrow">{architecture ? "Observed architecture relationships" : "Symbol graph"}</p>{items.map((item, index) => <div className="graph-row" key={index}><strong>{String(item.source_symbol || item.source_file || "Unknown source")}</strong><span>{String(item.edge_type || "RELATED_TO")}</span><strong>{String(item.target_symbol || item.target_file || "Unknown target")}</strong></div>)}</section>;
}

function Chat({ question, setQuestion, submit, answer }: { question: string; setQuestion: (value: string) => void; submit: (event: FormEvent) => void; answer: { text: string; evidence?: unknown[] } | null }) {
  return <section className="chat-view"><div className="chat-intro"><p className="eyebrow">Repository Chat</p><h2>Ask about the indexed repository.</h2><p className="muted">Answers come from repository retrieval and preserve the distinction between observed evidence and inference.</p></div>{answer ? <article className="chat-answer"><p>{answer.text}</p>{answer.evidence?.length ? <div className="evidence-list"><strong>Evidence</strong><pre>{JSON.stringify(answer.evidence, null, 2)}</pre></div> : null}</article> : <EmptyState>No question asked yet.</EmptyState>}<form className="chat-form" onSubmit={submit}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Where is authentication implemented?" /><button className="btn btn-primary" type="submit">Ask</button></form></section>;
}

function Runs({ status, run }: { status: RepositoryStatus | null; run: RunDetail | null }) {
  const stages = ["INGESTION", "PARSING", "KNOWLEDGE", "EMBEDDING", "GRAPH", "ANALYSIS"];
  return <section className="runs-view"><p className="eyebrow">Run {run?.run_id || status?.last_run_id || "not started"}</p>{stages.map((stage) => { const state = run?.lifecycle?.stages?.[stage]?.status || (run?.current_stage === stage ? "IN_PROGRESS" : "PENDING"); return <div className="stage-row" key={stage}><span className={`stage-dot stage-${state.toLowerCase()}`} /> <strong>{stage}</strong><span>{state.replaceAll("_", " ")}</span></div>; })}<p className="muted">Planner, retriever, generator, reviewer, and reflection details are shown here when the agent run persists those events.</p></section>;
}
