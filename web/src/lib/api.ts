export type ApiError = Error & { status?: number };

const API_ROOT = "/api";

export function getToken() {
  return localStorage.getItem("repointel:access-token");
}

export function setToken(token: string) {
  localStorage.setItem("repointel:access-token", token);
}

export function clearToken() {
  localStorage.removeItem("repointel:access-token");
}

export async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_ROOT}${path}`, { ...init, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401 && !path.startsWith("/auth/")) {
      clearToken();
      window.dispatchEvent(new CustomEvent("repointel:auth-required"));
    }
    const error = new Error(body.detail || `Request failed (${response.status})`) as ApiError;
    error.status = response.status;
    throw error;
  }
  return body as T;
}

export type RepositoryStatus = {
  repository_id: string;
  repo_url: string;
  status: string;
  branch?: string | null;
  head_commit?: string | null;
  archetype?: string | null;
  last_run_id?: string | null;
  artifact_counts?: Record<string, number>;
  recent_runs?: Array<{ run_id: string; status: string; mode: string; progress: number; files_parsed: number }>;
};

export type RunDetail = {
  run_id: string;
  status: string;
  mode?: string | null;
  current_stage?: string | null;
  progress?: number;
  error?: string | null;
  lifecycle?: { lifecycle_state?: string; stages?: Record<string, { status: string; counts?: Record<string, number> }> } | null;
  artifact_counts?: Record<string, number>;
};

export function createRepository(repo_url: string) {
  return apiRequest<{ repository_id: string; repo_url: string }>("/repositories", {
    method: "POST",
    body: JSON.stringify({ repo_url }),
  });
}

export function analyzeRepository(repository_id: string, repo_url: string) {
  return apiRequest<{ run_id: string; status: string }>(`/repositories/${repository_id}/analyze`, {
    method: "POST",
    body: JSON.stringify({ repo_url }),
  });
}

export function getRepositoryStatus(repository_id: string) {
  return apiRequest<RepositoryStatus>(`/repositories/${repository_id}/status`);
}

export function getRun(run_id: string) {
  return apiRequest<RunDetail>(`/runs/${run_id}`);
}

export function getArtifacts(repository_id: string) {
  return apiRequest<{ artifacts: Array<{ artifact_id: number; run_id: string; content: string; version: number; created_at: string; run_status: string }> }>(`/repositories/${repository_id}/artifacts`);
}

export function generateArtifact(repository_id: string, artifact_type: string = "README") {
  return apiRequest<{ repository_id: string; run_id: string; artifact_type: string; generated: boolean; warning?: string }>(`/repositories/${repository_id}/artifacts/generate?artifact_type=${encodeURIComponent(artifact_type)}`, {
    method: "POST",
  });
}

export function getDependencies(repository_id: string) {
  return apiRequest<{ dependencies: Array<{ package_name: string; ecosystem: string; version_spec: string; dependency_type: string }> }>(`/repositories/${repository_id}/dependencies`);
}

export function getSymbols(repository_id: string) {
  return apiRequest<{ symbols: Array<{ symbol_name: string; symbol_type: string; file_path: string; line_start: number; line_end: number; signature?: string }> }>(`/repositories/${repository_id}/symbols`);
}

export function getGraph(repository_id: string) {
  return apiRequest<{ neighbors: Array<{ source_symbol: string; source_file: string; edge_type: string; target_symbol: string; target_file: string }> }>(`/repositories/${repository_id}/graph`);
}

export type EvidenceItem = {
  file_path: string;
  line_start?: number;
  line_end?: number;
  symbol_name?: string;
  snippet?: string;
};

export type ProjectUnderstanding = {
  purpose: string;
  technologies: string[];
  architecture: string;
  key_components: Array<{ name: string; description?: string; file_path?: string; lines?: string }>;
  entry_points: Array<{ name: string; file_path?: string; lines?: string }>;
  evidence: EvidenceItem[];
  degraded?: boolean;
};

export function getUnderstanding(repository_id: string) {
  return apiRequest<ProjectUnderstanding>(`/repositories/${repository_id}/understanding`);
}

export function askRepository(repository_id: string, query: string) {
  return apiRequest<{ answer?: string; response?: string; evidence?: unknown[] }>(`/repositories/${repository_id}/chat`, {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

export function login(username: string, password: string) {
  return apiRequest<{ access_token: string; refresh_token: string; expires_in: number }>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function register(username: string, email: string, password: string) {
  return apiRequest<{ access_token: string; refresh_token: string; expires_in: number }>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, email, password }),
  });
}