from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from packages.agentic.tools import RepoIntelTools
from packages.knowledge.repository_facade import RepositoryFacade


@pytest.mark.asyncio
async def test_security_findings_facade_and_tool_return_persisted_data():
    row = SimpleNamespace(
        rule_id="python.lang.security.audit.exec-used",
        severity="HIGH",
        message="Avoid dynamic execution.",
        file_path="src/unsafe.py",
        line_start=12,
        line_end=12,
        finding_type="STATIC",
        source_endpoint="semgrep",
    )
    result = SimpleNamespace(
        scalars=lambda: SimpleNamespace(all=lambda: [row]),
    )
    session = SimpleNamespace(execute=AsyncMock(return_value=result))
    facade = RepositoryFacade(session, vector_store=object())

    findings = await facade.get_security_findings("repo-1")
    assert findings == [{
        "rule_id": "python.lang.security.audit.exec-used",
        "severity": "HIGH",
        "message": "Avoid dynamic execution.",
        "file_path": "src/unsafe.py",
        "line_start": 12,
        "line_end": 12,
        "finding_type": "STATIC",
        "source_endpoint": "semgrep",
    }]

    tool_result = await RepoIntelTools(facade).call("get_security_findings", "repo-1")
    assert tool_result == {"findings": findings, "count": 1}
    assert session.execute.await_count == 2