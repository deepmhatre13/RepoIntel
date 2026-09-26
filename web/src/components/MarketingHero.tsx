import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { analyzeRepository, createRepository } from "../lib/api";

/** Feature 1 — M7 Grain Drift + M1 Ink Rise in three waves. */
export function MarketingHero() {
  const navigate = useNavigate();
  const [repoUrl, setRepoUrl] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function startAnalysis() {
    setError("");
    setBusy(true);
    try {
      const repo = await createRepository(repoUrl.trim());
      const run = await analyzeRepository(repo.repository_id, repo.repo_url);
      localStorage.setItem("repointel:last-repository", repo.repository_id);
      navigate(`/app/repos/${repo.repository_id}/overview?run=${run.run_id}`);
    } catch (err) {
      setError((err as { status?: number }).status === 401
        ? "Sign in from the account menu to save and start this analysis."
        : err instanceof Error ? err.message : "Analysis could not be started.");
    } finally {
      setBusy(false);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    await startAnalysis();
  }

  return (
    <>
      <main className="landing-page">
        <section className="hero hero-intelligence">
          <div className="hero-inner">
            <p className="eyebrow">Repository intelligence</p>
            <h1>Understand what your repository <em>actually</em> says.</h1>
            <p className="hero-sub">Give RepoIntel a GitHub repository. It builds persistent knowledge from files, symbols, dependencies, and evidence, then turns that understanding into a README and repository-aware answers.</p>
            <form className="repo-entry" onSubmit={submit}>
              <input aria-label="GitHub repository URL" required value={repoUrl} onChange={(event) => setRepoUrl(event.target.value)} placeholder="https://github.com/owner/repository" type="url" />
              <button className="btn btn-primary" disabled={busy} type="submit">{busy ? "Starting analysis..." : "Analyze Repository"}</button>
            </form>
            {error ? <p className="form-error" role="alert">{error}</p> : null}
          </div>
        </section>
        <section className="intelligence-pillars">
          <article><p className="eyebrow">01 / Repository map</p><h2>Files, symbols, dependencies, and architecture.</h2><p>See the structure that was observed, with missing evidence left visible.</p></article>
          <article><p className="eyebrow">02 / README intelligence</p><h2>Documentation generated from repository evidence.</h2><p>Claims remain tied to the knowledge that produced them.</p></article>
          <article><p className="eyebrow">03 / Repository chat</p><h2>Ask questions and trace the answer back to source.</h2><p>Observed facts stay distinct from inference.</p></article>
        </section>
      </main>
    </>
  );
}
