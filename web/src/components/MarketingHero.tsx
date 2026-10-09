import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { analyzeRepository, createRepository } from "../lib/api";

export function MarketingHero() {
  const navigate = useNavigate();
  const [repoUrl, setRepoUrl] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function startAnalysis() {
    if (!repoUrl.trim()) return;
    setError("");
    setBusy(true);
    try {
      const repo = await createRepository(repoUrl.trim());
      const run = await analyzeRepository(repo.repository_id, repo.repo_url);
      localStorage.setItem("repointel:last-repository", repo.repository_id);
      navigate(`/app/repos/${repo.repository_id}/overview?run=${run.run_id}`);
    } catch (err) {
      setError(
        (err as { status?: number }).status === 401
          ? "Sign in from the account menu to start repository analysis."
          : err instanceof Error
          ? err.message
          : "Analysis could not be started."
      );
    } finally {
      setBusy(false);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    await startAnalysis();
  }

  return (
    <main className="landing-page">
      <section className="hero hero-intelligence">
        <div className="hero-inner">
          <p className="eyebrow">Repository Intelligence & Generation Platform</p>
          <h1>
            Understand any <em>GitHub repository</em>.
          </h1>
          <p className="hero-sub">
            Give RepoIntel a repository. Its analysis engine builds a structured understanding of the project, then AI agents explain, analyze and generate from the actual code.
          </p>
          <form className="repo-entry" onSubmit={submit}>
            <input
              aria-label="GitHub repository URL"
              required
              value={repoUrl}
              onChange={(event) => setRepoUrl(event.target.value)}
              placeholder="https://github.com/owner/repository"
              type="url"
            />
            <button className="btn btn-primary" disabled={busy} type="submit">
              {busy ? "Analyzing Repository..." : "Analyze Repository"}
            </button>
          </form>
          {error ? <p className="form-error" role="alert">{error}</p> : null}
        </div>
      </section>

      <section className="intelligence-pillars">
        <article>
          <p className="eyebrow">01 / Repository Understanding</p>
          <h2>Files, AST symbols, and graph architecture.</h2>
          <p>Constructs structured repository knowledge from files, classes, functions, and imports.</p>
        </article>
        <article>
          <p className="eyebrow">02 / LangGraph & Gemini Agents</p>
          <h2>Evidence-backed AI reasoning.</h2>
          <p>Specialized agents use Google Gemini to reason directly over verified repository facts.</p>
        </article>
        <article>
          <p className="eyebrow">03 / Provenance & Evidence</p>
          <h2>Every claim grounded in real code.</h2>
          <p>Answers and generated documentation cite exact file paths, symbol names, and line ranges.</p>
        </article>
      </section>
    </main>
  );
}
