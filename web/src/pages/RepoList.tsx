import { Link } from "react-router-dom";

export function RepoList() {
  const repositoryId = localStorage.getItem("repointel:last-repository");

  return (
    <main className="repo-page">
      <p className="eyebrow">Repository workspace</p>
      <h1>{repositoryId ? "Continue your repository investigation" : "No repository selected"}</h1>
      <p className="workspace-empty">{repositoryId ? "Your latest repository is persisted in the browser. Open its observed data or analyze another repository." : "Start with a GitHub repository URL to build a persistent knowledge workspace."}</p>
      <div className="cta-row">
        {repositoryId ? <Link className="btn btn-primary" to={`/app/repos/${repositoryId}/overview`}>Open repository</Link> : null}
        <Link className="btn btn-ghost" to="/">Analyze Repository</Link>
      </div>
    </main>
  );
}
