"""GitHub Health Provider for Project Health Intelligence.

The GitHubHealthProvider owns the boundary between Project Health and raw
GitHub REST data. The health engine never touches GitHub response shapes —
it consumes normalized :class:`ProviderData` records.

Responsibilities handled here:
* input normalization (``owner/repo`` or a full GitHub URL),
* 0 vs not-available: every domain reports ``status``
  (``AVAILABLE`` / ``ZERO`` / ``NOT_AVAILABLE`` / ``INSUFFICIENT_DATA`` /
  ``API_ERROR``) rather than silently substituting ``0``,
* authorization (optional token for higher rate limits, never exposed),
* timeout / retry-with-backoff per request,
* pagination bounded by a max page budget (rate-limit friendly),
* graceful partial responses when a sub-source fails.

The token is read solely from the shared config layer (``GITHUB_TOKEN``) and is
never sent anywhere other than the GitHub API and never logged.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from packages.shared.config import settings
from packages.shared.tracing import span
from packages.project_health.models import (
    CommitRecord,
    ContributorRecord,
    IssueRecord,
    ProviderData,
    PullRequestRecord,
    ReleaseRecord,
    RepositoryMetadata,
)

GITHUB_API = "https://api.github.com"
GITHUB_WEB = "https://github.com"


class GitHubHealthProviderError(Exception):
    """Base error for provider failures (never counts as a measured zero)."""


class RepositoryNotFoundError(GitHubHealthProviderError):
    pass


class RateLimitError(GitHubHealthProviderError):
    pass


def normalize_repository_input(value: str) -> str:
    """Normalize ``owner/repo`` or ``https://github.com/owner/repo`` → owner/repo.

    Raises ``ValueError`` for malformed inputs so the API can return a 400
    rather than attempting a pointless network call.
    """
    raw = (value or "").strip().strip("/")
    if not raw:
        raise ValueError("Repository input is empty.")
    # Strip an optional github.com/ prefix (with or without scheme) so that
    # "github.com/owner/repo" and "https://github.com/owner/repo" both work.
    if "github.com/" in raw:
        raw = raw.split("github.com/", 1)[1]
    if raw.startswith("http://") or raw.startswith("https://"):
        raise ValueError("Only github.com repository URLs are supported.")
    elif raw.startswith(("git@", "git://")):
        raise ValueError("git over SSH URLs are not supported; use owner/repository.")
    parts = raw.split("/")
    if len(parts) < 2:
        raise ValueError("Expected owner/repository (e.g. octocat/hello-world).")
    owner, repo = parts[0], parts[1]
    repo = repo.strip("/").strip()
    return f"{owner}/{repo}"


class GitHubHealthProvider:
    """Retrieve and normalize public GitHub repository health data."""

    def __init__(self, token: Optional[str] = None, max_pages: int = 5,
                 timeout: float = 15.0, max_retries: int = 2):
        settings.load_dotenv()
        # Token: explicit arg > environment. Never logged / never sent to Gemini.
        self._token = token if token is not None else settings.get("GITHUB_TOKEN")
        self.max_pages = max_pages
        self.timeout = timeout
        self.max_retries = max_retries

    # -- auth / request plumbing --------------------------------------------
    @property
    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/vnd.github+json",
             "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        """GET with bounded retries + backoff. Raises provider errors."""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.get(url, headers=self._headers, params=params,
                                    timeout=self.timeout)
            except requests.RequestException as exc:  # network/timeout
                last_err = exc
                time.sleep(0.5 * (attempt + 1))
                continue
            if resp.status_code == 403 and (resp.headers.get("X-RateLimit-Remaining") == "0"
                                            or "rate limit" in resp.text.lower()):
                reset = resp.headers.get("X-RateLimit-Reset")
                raise RateLimitError(
                    "GitHub rate limit exceeded" + (f" (resets {reset})" if reset else "")
                )
            if resp.status_code == 404:
                raise RepositoryNotFoundError(f"Repository not found: {url}")
            if resp.status_code >= 500:
                last_err = GitHubHealthProviderError(f"GitHub 5xx: {resp.status_code}")
                time.sleep(1.0 * (attempt + 1))
                continue
            if resp.status_code < 200 or resp.status_code >= 300:
                raise GitHubHealthProviderError(f"GitHub API error {resp.status_code}")
            return resp
        raise GitHubHealthProviderError(f"GitHub request failed after retries: {last_err}")

    def _paginate(self, path: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch `path` across paginated pages (bounded by max_pages).

        Returns a flat list of raw item dicts. A failed page beyond the first
        is tolerated (partial response) — the caller still sees whatever items
        were retrieved.
        """
        results: List[Dict[str, Any]] = []
        page = 1
        params = dict(params or {})
        params["per_page"] = 100
        while page <= self.max_pages:
            p = dict(params)
            p["page"] = page
            try:
                resp = self._get(f"{GITHUB_API}{path}", params=p)
            except (RepositoryNotFoundError, RateLimitError):
                raise
            except Exception:
                # Non-fatal pagination failure: stop fetching, keep what we have.
                break
            items = resp.json()
            if not isinstance(items, list) or not items:
                break
            results.extend(items)
            page += 1
            if len(items) < 100:
                break
        return results
# -- domain fetchers ----------------------------------------------------
    def fetch_repository(self, owner: str, repo: str) -> RepositoryMetadata:
        with span("github.fetch", resource="repository", owner=owner, repo=repo):
            resp = self._get(f"{GITHUB_API}/repos/{owner}/{repo}")
            d = resp.json()
        return RepositoryMetadata(
            owner=owner,
            name=repo,
            full_name=f"{owner}/{repo}",
            url=f"{GITHUB_WEB}/{owner}/{repo}",
            stars=_int_or_none(d.get("stargazers_count")),
            forks=_int_or_none(d.get("forks_count")),
            watchers=_int_or_none(d.get("subscribers_count")),
            open_issues=_int_or_none(d.get("open_issues_count")),
            created_at=_dt_or_none(d.get("created_at")),
            pushed_at=_dt_or_none(d.get("pushed_at")),
            default_branch=d.get("default_branch"),
            languages=_dict_or_empty(d.get("language")),
            license=(d.get("license") or {}).get("spdx_id"),
            archived=bool(d.get("archived", False)),
            size_kb=_int_or_none(d.get("size")),
            description=d.get("description"),
        )

    def fetch_commits(self, owner: str, repo: str) -> List[CommitRecord]:
        items = self._paginate(f"/repos/{owner}/{repo}/commits")
        out = []
        for it in items:
            commit = it.get("commit") or {}
            author = commit.get("author") or {}
            stats = it.get("stats") or {}
            out.append(
                CommitRecord(
                    sha=it.get("sha", ""),
                    author=author.get("name"),
                    committed_at=_dt_or_none(author.get("date")),
                    additions=_int_or_none(stats.get("additions")),
                    deletions=_int_or_none(stats.get("deletions")),
                    files_changed=_int_or_none(stats.get("total")),
                    message=(commit.get("message") or "")[:300],
                )
            )
        return out

    def fetch_pull_requests(self, owner: str, repo: str) -> List[PullRequestRecord]:
        items = self._paginate(f"/repos/{owner}/{repo}/pulls",
                               {"state": "all", "sort": "created", "direction": "desc"})
        out = []
        for it in items:
            merged = it.get("merged_at") is not None
            head = it.get("head") or {}
            out.append(
                PullRequestRecord(
                    number=_int_or_none(it.get("number")) or 0,
                    author=(it.get("user") or {}).get("login"),
                    opened_at=_dt_or_none(it.get("created_at")),
                    closed_at=None if merged else _dt_or_none(it.get("closed_at")),
                    merged_at=_dt_or_none(it.get("merged_at")),
                    state="merged" if merged else it.get("state", "open"),
                    first_response_at=None,  # NOT directly available on list endpoint
                    reviewer_count=len(it.get("requested_reviewers") or []),
                    comment_count=_int_or_none(it.get("comments")) or 0,
                    changed_files=_int_or_none(it.get("changed_files")),
                    additions=_int_or_none(it.get("additions")),
                    deletions=_int_or_none(it.get("deletions")),
                )
            )
        return out

    def fetch_issues(self, owner: str, repo: str) -> List[IssueRecord]:
        items = self._paginate(f"/repos/{owner}/{repo}/issues",
                               {"state": "all", "sort": "created", "direction": "desc"})
        out = []
        for it in items:
            if it.get("pull_request"):
                continue  # Issues endpoint returns PRs too; skip them.
            out.append(
                IssueRecord(
                    number=_int_or_none(it.get("number")) or 0,
                    author=(it.get("user") or {}).get("login"),
                    opened_at=_dt_or_none(it.get("created_at")),
                    closed_at=_dt_or_none(it.get("closed_at")),
                    state=it.get("state", "open"),
                    comment_count=_int_or_none(it.get("comments")) or 0,
                    labels=[lbl.get("name", "") for lbl in (it.get("labels") or [])],
                )
            )
        return out

    def fetch_releases(self, owner: str, repo: str) -> List[ReleaseRecord]:
        items = self._paginate(f"/repos/{owner}/{repo}/releases")
        return [
            ReleaseRecord(
                tag=it.get("tag_name", ""),
                published_at=_dt_or_none(it.get("published_at")),
                prerelease=bool(it.get("prerelease", False)),
                draft=bool(it.get("draft", False)),
            )
            for it in items
        ]

    def fetch_contributors(self, owner: str, repo: str) -> List[ContributorRecord]:
        items = self._paginate(f"/repos/{owner}/{repo}/contributors", {"anon": "false"})
        return [
            ContributorRecord(
                login=it.get("login", ""),
                contributions=_int_or_none(it.get("contributions")) or 0,
            )
            for it in items
        ]

    # -- top level ----------------------------------------------------------
    def fetch_project_health_data(self, owner: str, repo: str, *,
                                  include: Optional[List[str]] = None) -> ProviderData:
        """Fetch and normalize GitHub health data with per-domain status.

        ``include`` limits which domains to fetch (None = all). Every domain
        reports an explicit ``status`` so the engine knows whether ``0`` is a
        real zero or an unavailable source.
        """
        include = include or ["metadata", "commits", "pulls", "issues",
                              "releases", "contributors"]
        data = ProviderData(status={})
        # Use the model's own status dict so later mutations are reflected on
        # the returned object (Pydantic v2 copies dicts on validation).
        status = data.status

        def _default_meta() -> RepositoryMetadata:
            return RepositoryMetadata(
                owner=owner, name=repo, full_name=f"{owner}/{repo}",
                url=f"{GITHUB_WEB}/{owner}/{repo}",
            )

        if "metadata" in include:
            try:
                data.metadata = self.fetch_repository(owner, repo)
                status["metadata"] = "AVAILABLE"
            except RepositoryNotFoundError as exc:
                status["metadata"] = "NOT_AVAILABLE"
                data.metadata_error = str(exc)
            except RateLimitError:
                status["metadata"] = "API_ERROR"
            except GitHubHealthProviderError:
                status["metadata"] = "API_ERROR"
        if data.metadata is None:
            data.metadata = _default_meta()

        if "commits" in include:
            try:
                data.commits = self.fetch_commits(owner, repo)
                status["commits"] = "AVAILABLE" if data.commits else "ZERO"
            except (RepositoryNotFoundError, RateLimitError):
                status["commits"] = "API_ERROR"
            except Exception:
                status["commits"] = "API_ERROR"

        if "pulls" in include:
            try:
                prs = self.fetch_pull_requests(owner, repo)
                data.pull_requests = prs
                if data.metadata.open_prs is None:
                    data.metadata.open_prs = len([p for p in prs if p.state == "open"])
                if data.metadata.merged_prs is None:
                    data.metadata.merged_prs = len([p for p in prs if p.state == "merged"])
                status["pulls"] = "AVAILABLE" if prs else "ZERO"
            except Exception:
                status["pulls"] = "API_ERROR"

        if "issues" in include:
            try:
                issues = self.fetch_issues(owner, repo)
                data.issues = issues
                if data.metadata.open_issues is None:
                    data.metadata.open_issues = len([i for i in issues if i.state == "open"])
                if data.metadata.closed_issues is None:
                    data.metadata.closed_issues = len([i for i in issues if i.state == "closed"])
                status["issues"] = "AVAILABLE" if issues else "ZERO"
            except Exception:
                status["issues"] = "API_ERROR"

        if "releases" in include:
            try:
                data.releases = self.fetch_releases(owner, repo)
                status["releases"] = "AVAILABLE" if data.releases else "ZERO"
            except Exception:
                status["releases"] = "API_ERROR"

        if "contributors" in include:
            try:
                data.contributors = self.fetch_contributors(owner, repo)
                status["contributors"] = "AVAILABLE" if data.contributors else "ZERO"
            except Exception:
                status["contributors"] = "API_ERROR"

        return data


# -- small type helpers -----------------------------------------------------
def _int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _dt_or_none(v: Any) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, datetime):
        return v
    s = str(v).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            return datetime.strptime(str(v), "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc)
        except ValueError:
            return None


def _dict_or_empty(v: Any) -> Dict[str, int]:
    return v if isinstance(v, dict) else {}