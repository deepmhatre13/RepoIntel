import os
import json
from typing import Dict, List, Tuple
from packages.shared import RepositoryArchetype
from packages.ingestion.exceptions import MetadataExtractionError

MANIFEST_PATTERNS = {
    "package.json", "requirements.txt", "pyproject.toml", "setup.py",
    "go.mod", "Cargo.toml", "pom.xml", "build.gradle"
}

ENTRYPOINT_PATTERNS = {
    "main.py", "app.py", "manage.py", "index.js", "server.js", "main.go",
    "src/index.ts", "src/main.rs", "cmd/main.go"
}

CI_PATTERNS = {
    ".github/workflows", ".gitlab-ci.yml", "bitbucket-pipelines.yml", "circleci"
}

EXT_LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".cpp": "C++",
    ".c": "C",
    ".md": "Markdown",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML"
}

def extract_metadata(repo_path: str, filtered_files: List[str]) -> Tuple[Dict[str, int], List[str], List[str], List[str]]:
    languages: Dict[str, int] = {}
    manifest_files = []
    entrypoints = []
    ci_configs = []

    for rel_path in filtered_files:
        filename = os.path.basename(rel_path)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in EXT_LANGUAGE_MAP:
            lang = EXT_LANGUAGE_MAP[ext]
            languages[lang] = languages.get(lang, 0) + 1

        if filename in MANIFEST_PATTERNS or rel_path in MANIFEST_PATTERNS:
            manifest_files.append(rel_path)
            
        if filename in ENTRYPOINT_PATTERNS or rel_path in ENTRYPOINT_PATTERNS:
            entrypoints.append(rel_path)

        for ci in CI_PATTERNS:
            if ci in rel_path:
                ci_configs.append(rel_path)

    return languages, manifest_files, entrypoints, ci_configs

def detect_archetype(repo_path: str, filtered_files: List[str], manifest_files: List[str]) -> RepositoryArchetype:
    # 1. Monorepo detection
    if "turbo.json" in filtered_files or "lerna.json" in filtered_files or "pnpm-workspace.yaml" in filtered_files:
        return RepositoryArchetype.MONOREPO
    manifest_count = len([m for m in manifest_files if m in {"package.json", "requirements.txt", "go.mod", "Cargo.toml"}])
    if manifest_count > 2 or any(m.count("/") >= 2 for m in manifest_files):
        return RepositoryArchetype.MONOREPO

    # 2. CLI Tool detection
    pkg_json = os.path.join(repo_path, "package.json")
    if os.path.exists(pkg_json):
        try:
            from packages.parsing.decoding import decode_file_bytes
            with open(pkg_json, "rb") as f:
                raw_bytes = f.read()
            dec_res = decode_file_bytes(raw_bytes, "package.json")
            if dec_res.success:
                data = json.loads(dec_res.content)
                if "bin" in data:
                    return RepositoryArchetype.CLI_TOOL
        except Exception:
            pass

    # 3. Service detection
    if any(f in filtered_files for f in ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]) or \
       any("server" in f.lower() or "service" in f.lower() or "api" in f.lower() for f in filtered_files):
        return RepositoryArchetype.SERVICE

    # 4. Library detection
    if any(m in manifest_files for m in ["setup.py", "pyproject.toml", "Cargo.toml"]):
        return RepositoryArchetype.LIBRARY

    return RepositoryArchetype.UNKNOWN
