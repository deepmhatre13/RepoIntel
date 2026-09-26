import { FormEvent, useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { PageTurn } from "./components/PageTurn";
import { clearToken, getToken, login, register, setToken } from "./lib/api";

function AccountMenu({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const response = mode === "login"
        ? await login(username, password)
        : await register(username, email, password);
      setToken(response.access_token);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed.");
    } finally {
      setBusy(false);
    }
  }

  if (!open) return null;
  const authenticated = Boolean(getToken());
  return (
    <div className="account-popover" role="dialog" aria-label="Account">
      <div className="account-heading">
        <strong>Account</strong>
        <button className="icon-button account-close" type="button" onClick={onClose} aria-label="Close account">×</button>
      </div>
      {authenticated ? (
        <div className="account-actions">
          <p className="muted">Signed in to RepoIntel</p>
          <button className="account-link" type="button">Account settings</button>
          <button className="account-link" type="button" onClick={() => { clearToken(); onClose(); }}>Sign out</button>
        </div>
      ) : (
        <form className="account-form" onSubmit={submit}>
          <p className="muted">Sign in to save repository analysis and workspace state.</p>
          <input required value={username} onChange={(event) => setUsername(event.target.value)} placeholder="Username" autoComplete="username" />
          {mode === "register" ? <input required value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Email" type="email" autoComplete="email" /> : null}
          <input required minLength={8} value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Password" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} />
          {error ? <p className="form-error" role="alert">{error}</p> : null}
          <button className="btn btn-primary" type="submit" disabled={busy}>{busy ? "Working..." : mode === "login" ? "Sign in" : "Create account"}</button>
          <button className="account-link" type="button" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>{mode === "login" ? "Create account" : "Already have an account? Sign in"}</button>
          <div className="account-divider"><span>OR</span></div>
          <button className="btn btn-ghost" type="button" disabled>Continue with GitHub</button>
        </form>
      )}
    </div>
  );
}

export function App() {
  const [accountOpen, setAccountOpen] = useState(false);

  useEffect(() => {
    const openAccount = () => setAccountOpen(true);
    window.addEventListener("repointel:auth-required", openAccount);
    return () => window.removeEventListener("repointel:auth-required", openAccount);
  }, []);

  return (
    <>
      <nav className="shell-nav">
        <NavLink to="/" className="wordmark">
          Repo<em>Intel</em>
        </NavLink>
        <div className="nav-links">
          <NavLink to="/" end>Analyze Repository</NavLink>
          <NavLink to="/app">Workspace</NavLink>
          <NavLink to="/health">Advanced Analysis</NavLink>
        </div>
        <button className="icon-button account-button" type="button" onClick={() => setAccountOpen((open) => !open)} aria-label="Open account" aria-expanded={accountOpen}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="8" r="3.5" /><path d="M4.5 20c.8-3.4 3.2-5 7.5-5s6.7 1.6 7.5 5" /></svg>
        </button>
        <AccountMenu open={accountOpen} onClose={() => setAccountOpen(false)} />
      </nav>
      <PageTurn>
        <Outlet />
      </PageTurn>
    </>
  );
}
