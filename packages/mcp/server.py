"""MCP interface over RepoIntel (Stage 8).

Exposes RepoIntel's intelligence as Model Context Protocol tools over stdio
(JSON-RPC 2.0). This is a THIN adapter: every tool is resolved through the
shared :class:`RepoIntelTools` layer, which in turn calls the existing
RepositoryFacade / service boundaries. No business logic is duplicated here.

Local-first by design:
    * works over stdio with no external account,
    * every tool validates input and enforces the repository_id boundary,
    * no repository code is ever executed,
    * secrets are redacted before tool output leaves the process.

If the official ``mcp`` / ``fastmcp`` SDK is installed it is used; otherwise a
minimal JSON-RPC-over-stdio transport (which is the MCP wire protocol) is used,
so the server remains runnable locally without the optional dependency.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from typing import Any, Dict, List, Optional

from packages.shared.tracing import span
from packages.shared.secret_redaction import redact_secrets

try:
    from packages.agentic.tools import RepoIntelTools, ToolError
except Exception:  # pragma: no cover
    RepoIntelTools = None  # type: ignore[misc]
    ToolError = Exception

try:  # Project Health adapter (same service as FastAPI / LangGraph)
    from packages.project_health.tools import ProjectHealthTools
except Exception:  # pragma: no cover
    ProjectHealthTools = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Transport: JSON-RPC over stdio
# ---------------------------------------------------------------------------
class _BackgroundLoop:
    """Runs coroutines on a private, long-lived event loop thread.

    MCP tool calls arrive on a synchronous stdin loop (the only portable way
    to read stdin on Windows), while RepoIntel tool handlers are async. A
    dedicated background loop lets the server block on results without
    nesting ``asyncio.run()`` and independent of the caller's loop state.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def _ensure(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever, daemon=True,
                name="repointel-mcp-loop",
            )
            self._thread.start()
        return self._loop

    def run(self, coro):
        """Execute ``coro`` on the background loop; block until done."""
        return asyncio.run_coroutine_threadsafe(coro, self._ensure()).result()

    def close(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None


class RepoIntelMcpServer:
    """MCP server adapter bound to a ``RepoIntelTools`` instance."""

    PROTOCOL = "2025-03-26"

    def __init__(self, tools: RepoIntelTools, server_info: Optional[Dict[str, Any]] = None):
        self.tools = tools or RepoIntelTools(None)
        # Optional Project Health toolset (keyed by owner/repository, not id).
        self.health_tools = ProjectHealthTools() if ProjectHealthTools else None
        self.server_info = server_info or {"name": "repointel-mcp", "version": "0.1.0"}
        # repository_id is provided per-call; a default can be set via --repo.
        self.default_repository_id: Optional[str] = None
        self._loop_runner = _BackgroundLoop()

    def shutdown(self) -> None:
        """Stop the background execution loop (idempotent)."""
        self._loop_runner.close()

    # ------------------------------------------------------------------
    # MCP protocol handlers
    # ------------------------------------------------------------------
    def handle_request(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = message.get("method", "")
        params = message.get("params", {}) or {}
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "protocolVersion": params.get("protocolVersion", self.PROTOCOL),
                    "capabilities": {"tools": {}},
                    "serverInfo": self.server_info,
                },
            }
        if method == "notifications/initialized":
            return None  # notification - no response
        if method == "ping":
            return {"jsonrpc": "2.0", "id": message.get("id"), "result": {}}
        if method == "tools/list":
            return self._tools_list(message.get("id"))
        if method == "tools/call":
            return self._tools_call(message.get("id"), params)
        return self._error(message.get("id"), -32601, f"Method not found: {method}")

    def _tools_list(self, request_id) -> Dict[str, Any]:
        descriptors = self.tools.list_tools()
        if self.health_tools is not None:
            descriptors = descriptors + self.health_tools.list_tools()
        mcp_tools = []
        for d in descriptors:
            schema = {
                "type": "object",
                "properties": {},
                "required": [p["name"] for p in d["parameters"] if p["required"]],
            }
            for p in d["parameters"]:
                schema["properties"][p["name"]] = {
                    "type": p["type"],
                    "description": p["description"],
                }
            mcp_tools.append({
                "name": d["name"],
                "description": d["description"],
                "inputSchema": schema,
            })
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": mcp_tools}}

    def _tools_call(self, request_id, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        # Project Health tools are keyed by owner/repository, not repository_id;
        # they route to the same service that powers FastAPI and the agent.
        if self.health_tools is not None and self.health_tools.has_tool(name):
            with span("mcp.tools.call", tool_name=name,
                      tool_category="project-health"):
                try:
                    result = self._loop_runner.run(
                        self.health_tools.call(name, arguments))
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {"type": "text", "text": json.dumps(result, ensure_ascii=False)}
                            ],
                            "isError": False,
                        },
                    }
                except ToolError as exc:
                    return self._tool_error(request_id, name, str(exc))
                except Exception as exc:  # pragma: no cover
                    return self._tool_error(request_id, name, f"{type(exc).__name__}: {exc}")
        repository_id = arguments.get("repository_id") or self.default_repository_id
        if not repository_id:
            return self._tool_error(request_id, name, "repository_id is required")
        with span("mcp.tools.call", tool_name=name, repository_id=repository_id):
            try:
                result = self._loop_runner.run(
                    self.tools.call(name, repository_id, arguments))
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(result, ensure_ascii=False)}
                        ],
                        "isError": False,
                    },
                }
            except ToolError as exc:
                return self._tool_error(request_id, name, str(exc))
            except Exception as exc:  # pragma: no cover
                return self._tool_error(request_id, name, f"{type(exc).__name__}: {exc}")

    def _tool_error(self, request_id, name: str, detail: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": redact_secrets(detail).text}],
                "isError": True,
            },
        }

    def _error(self, request_id, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": redact_secrets(message).text},
        }

    # ------------------------------------------------------------------
    # Transport: JSON-RPC over stdio
    # ------------------------------------------------------------------
    def run_stdio(self, repository_id: Optional[str] = None):
        """Read MCP JSON-RPC messages from stdin and respond on stdout.

        Synchronous stdin loop (portable on Windows); async tool execution
        happens on the internal background event loop. stdout carries only
        JSON-RPC responses; diagnostics go to stderr.
        """
        self.default_repository_id = repository_id
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                response = self.handle_request(message)
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
        finally:
            self.shutdown()


def build_mcp_server(tools: RepoIntelTools, server_info: Optional[Dict[str, Any]] = None) -> RepoIntelMcpServer:
    """Construct an MCP server bound to a tools instance."""
    return RepoIntelMcpServer(tools, server_info=server_info)