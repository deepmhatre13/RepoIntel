import os
import json
import re
from dataclasses import dataclass, field
from typing import List

@dataclass
class ParsedDependency:
    package_name: str
    ecosystem: str  # pypi, npm, cargo, go
    version_spec: str
    dependency_type: str  # direct, dev, peer

@dataclass
class ManifestExtractionResult:
    dependencies: List[ParsedDependency] = field(default_factory=list)
    env_vars: List[str] = field(default_factory=list)

ENV_PATTERN_PY = re.compile(r'os\.(?:environ\.get|getenv|environ\[[\'"])([A-Za-z0-9_]+)')
ENV_PATTERN_JS = re.compile(r'process\.env\.([A-Za-z0-9_]+)')

def extract_dependencies_from_manifest(file_path: str, content: str) -> List[ParsedDependency]:
    filename = os.path.basename(file_path)
    deps = []

    # package.json
    if filename == "package.json":
        try:
            data = json.loads(content)
            for pkg, ver in data.get("dependencies", {}).items():
                deps.append(ParsedDependency(package_name=pkg, ecosystem="npm", version_spec=str(ver), dependency_type="direct"))
            for pkg, ver in data.get("devDependencies", {}).items():
                deps.append(ParsedDependency(package_name=pkg, ecosystem="npm", version_spec=str(ver), dependency_type="dev"))
        except Exception:
            pass

    # requirements.txt
    elif filename in ("requirements.txt", "requirements-dev.txt"):
        dep_type = "dev" if "dev" in filename else "direct"
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                match = re.split(r'[==|>=|<=|~=|>|<]', line, maxsplit=1)
                pkg = match[0].strip()
                ver = match[1].strip() if len(match) > 1 else "*"
                deps.append(ParsedDependency(package_name=pkg, ecosystem="pypi", version_spec=ver, dependency_type=dep_type))

    # pyproject.toml (simple extraction)
    elif filename == "pyproject.toml":
        in_deps = False
        for line in content.splitlines():
            line = line.strip()
            if "dependencies =" in line or "[tool.poetry.dependencies]" in line:
                in_deps = True
                continue
            if in_deps:
                if line.startswith("[") or not line:
                    in_deps = False
                    continue
                if "=" in line or '"' in line or "'" in line:
                    parts = line.split("=", 1)
                    pkg = parts[0].strip(' "\'[],')
                    ver = parts[1].strip(' "\'[],') if len(parts) > 1 else "*"
                    if pkg:
                        deps.append(ParsedDependency(package_name=pkg, ecosystem="pypi", version_spec=ver, dependency_type="direct"))

    # go.mod
    elif filename == "go.mod":
        in_require = False
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("require ("):
                in_require = True
                continue
            if in_require:
                if line.startswith(")"):
                    in_require = False
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    deps.append(ParsedDependency(package_name=parts[0], ecosystem="go", version_spec=parts[1], dependency_type="direct"))

    return deps

def extract_env_vars_from_code(content: str) -> List[str]:
    env_vars = set()
    for match in ENV_PATTERN_PY.findall(content):
        env_vars.add(match)
    for match in ENV_PATTERN_JS.findall(content):
        env_vars.add(match)
    return list(env_vars)
