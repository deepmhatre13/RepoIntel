import re
from typing import Any, Dict, List, Optional

from packages.agentic.state.schema import ArtifactSection, EvidenceCitation


_MANAGERS = (
    ("uv.lock", "uv"),
    ("poetry.lock", "poetry"),
    ("pipfile.lock", "pipenv"),
    ("requirements.txt", "pip"),
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("package-lock.json", "npm"),
    ("go.mod", "go"),
    ("cargo.lock", "cargo"),
)
_TEST_COMMANDS = (
    re.compile(r"(?m)^\s*(?:-\s*)?(?:run:\s*)?(python(?:3)? -m pytest(?:\s+[-\w./]+)*)\s*$"),
    re.compile(r"(?m)^\s*(?:-\s*)?(?:run:\s*)?(pytest(?:\s+[-\w./]+)*)\s*$"),
    re.compile(r"(?m)^\s*(?:-\s*)?(?:run:\s*)?(npm test)\s*$"),
    re.compile(r"(?m)^\s*(?:-\s*)?(?:run:\s*)?(pnpm test)\s*$"),
    re.compile(r"(?m)^\s*(?:-\s*)?(?:run:\s*)?(yarn test)\s*$"),
    re.compile(r"(?m)^\s*(?:-\s*)?(?:run:\s*)?(go test(?:\s+[-\w./]+)*)\s*$"),
    re.compile(r"(?m)^\s*(?:-\s*)?(?:run:\s*)?(cargo test(?:\s+[-\w./]+)*)\s*$"),
)
_LICENSE_PATTERNS = (
    (re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9.+-]+)"), 1),
    (re.compile(r"\bMIT License\b", re.IGNORECASE), "MIT"),
    (re.compile(r"Apache License(?:,? Version)?\s*2\.0", re.IGNORECASE), "Apache-2.0"),
    (re.compile(r"\bBSD 3-Clause License\b", re.IGNORECASE), "BSD-3-Clause"),
)


def _citation(path: str, chunk: Dict[str, Any]) -> EvidenceCitation:
    return EvidenceCitation(
        file_path=path,
        line_start=chunk["line_start"],
        line_end=chunk["line_end"],
        content=chunk["content"],
    )


def generate_grounded_quickstart(facts: Dict[str, Any]) -> Optional[ArtifactSection]:
    files: Dict[str, List[Dict[str, Any]]] = facts.get("files", {})
    citations: List[EvidenceCitation] = []
    lines = ["Repository setup and commands verified from indexed files:"]

    known_paths = set(facts.get("file_paths", []))
    entrypoints = [
        path for path in facts.get("entrypoints", [])
        if path in known_paths and files.get(path)
    ]
    if entrypoints:
        lines.append("\n### Entry points")
        for path in sorted(set(entrypoints)):
            lines.append(f"- `{path}`")
            citations.append(_citation(path, files[path][0]))

    manager_match = None
    for manifest_name, manager in _MANAGERS:
        matching_path = next((
            path for path in known_paths
            if path.replace("\\", "/").rsplit("/", 1)[-1].casefold() == manifest_name
            and files.get(path)
        ), None)
        if matching_path:
            manager_match = (matching_path, manager)
            break
    if manager_match:
        path, manager = manager_match
        lines.append("\n### Package manager")
        lines.append(f"- `{manager}` (detected from `{path}`)")
        citations.append(_citation(path, files[path][0]))

    test_files = {
        path: chunks
        for path, chunks in files.items()
        if path.casefold().endswith((
            "pyproject.toml", "pytest.ini", "tox.ini", "makefile", "justfile",
        )) or ".github/workflows/" in path.replace("\\", "/").casefold()
    }
    discovered_commands = []
    for path, chunks in sorted(test_files.items()):
        for chunk in chunks:
            content = chunk.get("content", "")
            for pattern in _TEST_COMMANDS:
                for match in pattern.finditer(content):
                    command = match.group(1).strip()
                    if command and command not in [item[0] for item in discovered_commands]:
                        discovered_commands.append((command, path, chunk))
    if discovered_commands:
        lines.append("\n### Test command")
        for command, path, chunk in discovered_commands:
            lines.append(f"- `{command}` (found in `{path}`)")
            citations.append(_citation(path, chunk))

    license_files = {
        path: chunks for path, chunks in files.items()
        if path.replace("\\", "/").rsplit("/", 1)[-1].casefold().startswith(
            ("license", "copying")
        )
    }
    license_match = None
    for path, chunks in sorted(license_files.items()):
        for chunk in chunks:
            for pattern, fixed_value in _LICENSE_PATTERNS:
                match = pattern.search(chunk.get("content", ""))
                if match:
                    license_name = (
                        match.group(1) if fixed_value == 1 else fixed_value
                    )
                    license_match = (license_name, path, chunk)
                    break
            if license_match:
                break
        if license_match:
            break
    if license_match:
        license_name, path, chunk = license_match
        lines.append("\n### License")
        lines.append(f"- License: {license_name} (detected in `{path}`)")
        citations.append(_citation(path, chunk))

    if not citations:
        return None
    return ArtifactSection(
        section_title="Quickstart",
        content="\n".join(lines),
        citations=citations,
    )