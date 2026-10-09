import { FormEvent, ReactNode, useEffect, useState } from "react";
import { NavLink, useParams, useSearchParams } from "react-router-dom";
import {
  askRepository,
  generateArtifact,
  getArtifacts,
  getDependencies,
  getGraph,
  getRepositoryStatus,
  getRun,
  getSymbols,
  getUnderstanding,
  type ProjectUnderstanding,
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
  const [understanding, setUnderstanding] = useState<ProjectUnderstanding | null>(null);
  const [content, setContent] = useState<unknown>(null);
  const [error, setError] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<{
    text: string;
    confidence?: string;
    evidence?: Array<Record<string, unknown>>;
    degraded?: boolean;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);

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

        if (activeSection === "overview") {
          try {
            const und = await getUnderstanding(repositoryId);
            if (!cancelled) setUnderstanding(und);
          } catch {
            // Understanding will populate when available
          }
        }

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
    if (!repositoryId || !run || ["COMPLETE", "FAILED", "PARTIAL", "READY", "KNOWLEDGE_READY", "AGENT_READY"].includes(run.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await getRun(run.run_id);
        setRun(next);
        if (["COMPLETE", "READY", "AGENT_READY"].includes(next.status)) {
          setStatus(await getRepositoryStatus(repositoryId));
          setUnderstanding(await getUnderstanding(repositoryId));
        }
      } catch {
        // Next tick
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
      setAnswer({
        text: (result as { answer?: string; response?: string }).answer || (result as { response?: string }).response || "No answer was returned.",
        confidence: (result as { confidence?: string }).confidence || "HIGH",
        evidence: (result as { evidence?: Array<Record<string, unknown>> }).evidence || [],
        degraded: (result as { degraded?: boolean }).degraded || false,
      });
      setQuestion("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat is unavailable.");
    }
  }

  async function triggerGenerateReadme() {
    if (!repositoryId) return;
    setGenerating(true);
    setError("");
    try {
      await generateArtifact(repositoryId, "README");
      const res = await getArtifacts(repositoryId);
      setContent(res.artifacts[0] || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "README generation failed.");
    } finally {
      setGenerating(false);
    }
  }

  if (!repositoryId) return <EmptyState>No repository selected.</EmptyState>;
  const repoName = status ? formatRepository(status.repo_url) : "Repository";

  return (
    <div className="workspace-shell">
      <aside className="workspace-sidebar">
        <div className="workspace-context">
          <p className="eyebrow">Repository</p>
          <strong>{repoName}</strong>
          <span>{status?.archetype || "Analysis pending"}</span>
        </div>
        <nav aria-label="Repository workspace">
          {sections.map((item) => (
            <NavLink
              key={item.id}
              to={`/app/repos/${repositoryId}/${item.id}${searchParams.get("run") ? `?run=${searchParams.get("run")}` : ""}`}
              className={({ isActive }) => (isActive ? "workspace-nav active" : "workspace-nav")}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="workspace-advanced">
          <p className="eyebrow">Advanced Analysis</p>
          <NavLink to={`/health?repository=${repositoryId}`}>Project Health</NavLink>
          <NavLink to={`/health/compare?repository=${repositoryId}`}>Compare Projects</NavLink>
        </div>
      </aside>
      <main className="workspace-main">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">{activeSection === "runs" ? "Agent execution" : "Repository Intelligence"}</p>
            <h1>{activeSection === "overview" ? repoName : sections.find((item) => item.id === activeSection)?.label}</h1>
          </div>
          <span className={`status-mark status-${(status?.status || run?.status || "pending").toLowerCase()}`}>
            {status?.status || run?.status || "PENDING"}
          </span>
        </header>
        {error ? <p className="form-error" role="alert">{error}</p> : null}
        {loading && !status ? <EmptyState>Loading observed repository intelligence...</EmptyState> : null}
        {activeSection === "overview" ? <Overview repositoryId={repositoryId} status={status} run={run} understanding={understanding} /> : null}
        {activeSection === "readme" ? <Readme artifact={content as { content: string; version: number; created_at: string } | null} generating={generating} onGenerate={triggerGenerateReadme} /> : null}
        {activeSection === "dependencies" ? <DataList title="Dependencies" items={content as Array<Record<string, unknown>> | null} fields={["package_name", "ecosystem", "version_spec", "dependency_type"]} /> : null}
        {activeSection === "symbols" ? <DataList title="Symbols" items={content as Array<Record<string, unknown>> | null} fields={["symbol_name", "symbol_type", "file_path", "line_start", "line_end"]} /> : null}
        {activeSection === "graph" || activeSection === "architecture" ? <GraphView items={content as Array<Record<string, unknown>> | null} architecture={activeSection === "architecture"} /> : null}
        {activeSection === "chat" ? <Chat question={question} setQuestion={setQuestion} submit={submitQuestion} answer={answer} archetype={status?.archetype} /> : null}
        {activeSection === "runs" ? <Runs status={status} run={run} /> : null}
      </main>
    </div>
  );
}

function WorkflowTimeline({ runState }: { runState?: string }) {
  const current = (runState || "PENDING").toUpperCase();
  const stages = [
    { key: "ANALYSIS", label: "Repository Analysis" },
    { key: "UNDERSTANDING", label: "Repository Understanding" },
    { key: "RETRIEVAL", label: "Evidence Retrieval" },
    { key: "GEMINI", label: "Gemini Analyst" },
    { key: "REVIEW", label: "Reviewer" },
    { key: "RESULT", label: "Result" },
  ];

  function getStageStatus(index: number) {
    if (["COMPLETE", "READY", "KNOWLEDGE_READY", "AGENT_READY"].includes(current)) return "completed";
    if (current === "FAILED") return index === 0 ? "failed" : "pending";
    if (current === "PARSING" || current === "SCANNING" || current === "CLONING") return index === 0 ? "running" : "pending";
    if (current === "INDEXING" || current === "GRAPH_BUILDING") return index <= 1 ? "completed" : index === 2 ? "running" : "pending";
    return index === 0 ? "completed" : index <= 2 ? "running" : "pending";
  }

  return (
    <div className="agent-timeline-card">
      <h3>
        <span className="signal-pulse" />
        LangGraph Agentic Pipeline Execution
      </h3>
      <div className="timeline-steps">
        {stages.map((stg, i) => {
          const st = getStageStatus(i);
          return (
            <div key={stg.key} className={`timeline-step ${st}`}>
              <div className="step-indicator">
                {st === "completed" ? "✓" : st === "running" ? "●" : st === "failed" ? "✕" : "○"}
              </div>
              <span className="step-label">{stg.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Overview({
  repositoryId,
  status,
  run,
  understanding,
}: {
  repositoryId: string;
  status: RepositoryStatus | null;
  run: RunDetail | null;
  understanding: ProjectUnderstanding | null;
}) {
  return (
    <section className="understanding-container">
      <WorkflowTimeline runState={run?.current_stage || run?.status || status?.status} />

      <article className="understanding-card">
        <h2>AI Project Understanding</h2>

        <h3>Project Purpose</h3>
        <p>{understanding?.purpose || "Analyzing repository purpose..."}</p>

        <h3>Primary Technologies</h3>
        <div className="tech-tags">
          {understanding?.technologies?.map((tech, i) => (
            <span key={i} className="tech-tag">
              {tech}
            </span>
          )) || <span className="tech-tag">{status?.archetype || "Detecting..."}</span>}
        </div>

        <h3>Architecture Overview</h3>
        <p>{understanding?.architecture || "Building architectural graph..."}</p>

        <h3>Key Components</h3>
        {understanding?.key_components?.length ? (
          <div className="tech-tags">
            {understanding.key_components.map((comp, i) => (
              <span key={i} className="tech-tag" title={comp.description}>
                {comp.name} ({comp.file_path || "core"})
              </span>
            ))}
          </div>
        ) : (
          <p className="muted">Extracting symbols and components...</p>
        )}

        <h3>Entry Points</h3>
        {understanding?.entry_points?.length ? (
          <div className="tech-tags">
            {understanding.entry_points.map((ep, i) => (
              <span key={i} className="tech-tag">
                {ep.name || ep.file_path} (L{ep.lines || "1"})
              </span>
            ))}
          </div>
        ) : (
          <p className="muted">Discovering entry points...</p>
        )}

        {understanding?.evidence?.length ? (
          <div className="evidence-section">
            <div className="evidence-title">Repository Evidence ({understanding.evidence.length} sources)</div>
            <div className="evidence-grid">
              {understanding.evidence.map((ev, i) => (
                <div key={i} className="evidence-chip">
                  <div className="evidence-path">{ev.file_path}</div>
                  <div className="evidence-lines">
                    lines {ev.line_start || 1}–{ev.line_end || 50} {ev.symbol_name ? `· ${ev.symbol_name}` : ""}
                  </div>
                  {ev.snippet ? <div className="evidence-snippet">{ev.snippet}</div> : null}
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div className="action-bar">
          <NavLink to={`/app/repos/${repositoryId}/chat`} className="btn btn-primary">
            Ask RepoIntel
          </NavLink>
          <NavLink to={`/app/repos/${repositoryId}/readme`} className="btn btn-secondary">
            Generate README
          </NavLink>
          <NavLink to={`/app/repos/${repositoryId}/architecture`} className="btn btn-secondary">
            Explore Architecture
          </NavLink>
          <NavLink to={`/app/repos/${repositoryId}/symbols`} className="btn btn-secondary">
            Explore Code
          </NavLink>
        </div>
      </article>
    </section>
  );
}

function Readme({
  artifact,
  generating,
  onGenerate,
}: {
  artifact: { content: string; version: number; created_at: string } | null;
  generating: boolean;
  onGenerate: () => void;
}) {
  return (
    <section className="artifact-view">
      <div className="artifact-header">
        <div>
          <h2 style={{ margin: 0, fontSize: "1.25rem" }}>Repository README</h2>
          <p className="artifact-meta" style={{ margin: "4px 0 0 0" }}>
            {artifact
              ? `Generated from repository evidence · version ${artifact.version} · ${new Date(artifact.created_at).toLocaleString()}`
              : "No generated README artifact is present."}
          </p>
        </div>
        <button className="btn btn-primary" onClick={onGenerate} disabled={generating}>
          {generating ? "Generating README..." : artifact ? "Re-generate README" : "Generate README"}
        </button>
      </div>
      {artifact ? (
        <pre className="artifact-body">{artifact.content}</pre>
      ) : (
        <EmptyState>Click 'Generate README' to run the LangGraph artifact generator over indexed code evidence.</EmptyState>
      )}
    </section>
  );
}

function DataList({ title, items, fields }: { title: string; items: Array<Record<string, unknown>> | null; fields: string[] }) {
  if (!items?.length) return <EmptyState>No {title.toLowerCase()} were discovered in the indexed repository.</EmptyState>;
  return <section className="data-table"><h2>{title}</h2>{items.map((item, index) => <div className="data-row" key={`${String(item[fields[0]])}-${index}`}>{fields.map((field) => <span key={field}><small>{field.replaceAll("_", " ")}</small>{String(item[field] ?? "Not available")}</span>)}</div>)}</section>;
}

function GraphView({ items, architecture }: { items: Array<Record<string, unknown>> | null; architecture: boolean }) {
  if (!items?.length) return <EmptyState>No graph relationships are available for this repository yet.</EmptyState>;
  return <section className="data-table"><p className="eyebrow">{architecture ? "Observed architecture relationships" : "Symbol graph"}</p>{items.map((item, index) => <div className="graph-row" key={index}><strong>{String(item.source_symbol || item.source_file || "Unknown source")}</strong><span>{String(item.edge_type || "RELATED_TO")}</span><strong>{String(item.target_symbol || item.target_file || "Unknown target")}</strong></div>)}</section>;
}

function Chat({
  question,
  setQuestion,
  submit,
  answer,
  archetype,
}: {
  question: string;
  setQuestion: (value: string) => void;
  submit: (event: FormEvent) => void;
  answer: {
    text: string;
    confidence?: string;
    evidence?: Array<Record<string, unknown>>;
    degraded?: boolean;
  } | null;
  archetype?: string | null;
}) {
  const suggested = [
    "Where is entry point implemented?",
    "What are the main AST symbols?",
    "How are dependencies configured?",
    "What is the repository structure?",
  ];

  return (
    <section className="chat-view">
      <div className="chat-intro">
        <p className="eyebrow">Evidence-Backed Repository Q&A</p>
        <h2>Ask RepoIntel about this project</h2>
        <p className="muted">
          Answers use Google Gemini over retrieved evidence from {archetype ? `the ${archetype} codebase` : "the indexed repository"}. Facts cite exact file paths and line ranges.
        </p>
      </div>

      <div className="suggested-prompts">
        {suggested.map((prompt) => (
          <button key={prompt} className="prompt-chip" type="button" onClick={() => setQuestion(prompt)}>
            {prompt}
          </button>
        ))}
      </div>

      <form className="chat-form" onSubmit={submit}>
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask an engineering question about the codebase..."
        />
        <button className="btn btn-primary" type="submit">
          Ask
        </button>
      </form>

      {answer ? (
        <article className="chat-answer-card">
          <div className="answer-header">
            <strong>RepoIntel Gemini Response</strong>
            <span className={`status-mark status-${(answer.confidence || "high").toLowerCase()}`}>
              {answer.confidence || "HIGH"} CONFIDENCE
            </span>
          </div>

          <div className="answer-text">{answer.text}</div>

          {answer.evidence?.length ? (
            <div className="evidence-section">
              <div className="evidence-title">Retrieved Grounding Evidence ({answer.evidence.length} sources)</div>
              <div className="evidence-grid">
                {answer.evidence.map((ev, i) => (
                  <div key={i} className="evidence-chip">
                    <div className="evidence-path">{String(ev.file_path || "code")}</div>
                    <div className="evidence-lines">
                      lines {String(ev.line_start || 1)}–{String(ev.line_end || 50)} {ev.symbol_name ? `· ${String(ev.symbol_name)}` : ""}
                    </div>
                    {ev.snippet ? <div className="evidence-snippet">{String(ev.snippet)}</div> : null}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="form-error" style={{ background: "var(--warning-bg)", color: "var(--warning)", borderColor: "var(--warning)" }}>
              No supporting code evidence was retrieved for this question.
            </div>
          )}
        </article>
      ) : (
        <EmptyState>Ask a question or select a suggested prompt to explore the codebase.</EmptyState>
      )}
    </section>
  );
}

function Runs({ status, run }: { status: RepositoryStatus | null; run: RunDetail | null }) {
  const runs = status?.recent_runs || (run ? [run] : []);
  if (!runs.length) return <EmptyState>No agent runs recorded for this repository.</EmptyState>;
  return <section className="data-table"><h2>Recorded Runs</h2>{runs.map((r) => <div className="data-row" key={r.run_id}><span><small>Run ID</small>{r.run_id}</span><span><small>Status</small>{r.status}</span><span><small>Mode</small>{r.mode || "FULL"}</span><span><small>Progress</small>{Math.round((r.progress || 0) * 100)}%</span></div>)}</section>;
}
