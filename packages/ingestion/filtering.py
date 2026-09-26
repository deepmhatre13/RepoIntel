import os
import pathspec

from packages.ingestion.exceptions import FilteringError

IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".idea", ".vscode", ".next", "coverage", ".pytest_cache", ".mypy_cache"
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf", ".zip", ".tar",
    ".gz", ".7z", ".exe", ".dll", ".so", ".dylib", ".pyc", ".pyo", ".db",
    ".sqlite", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3"
}

from packages.parsing.decoding import decode_file_bytes

def load_gitignore_spec(repo_path: str) -> pathspec.PathSpec:
    gitignore_path = os.path.join(repo_path, ".gitignore")
    patterns = []
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, "rb") as f:
                raw_bytes = f.read()
            dec_res = decode_file_bytes(raw_bytes, ".gitignore")
            if dec_res.success:
                patterns = dec_res.content.splitlines(keepends=True)
        except Exception:
            pass
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

def filter_repository(repo_path: str) -> list[str]:
    try:
        spec = load_gitignore_spec(repo_path)
    except Exception as exc:
        raise FilteringError(f"Failed to load .gitignore spec: {exc}")
    allowed_files = []

    for root, dirs, files in os.walk(repo_path):
        # Prune ignored directory names in-place
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
        
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, repo_path).replace("\\", "/")
            
            ext = os.path.splitext(file)[1].lower()
            if ext in BINARY_EXTENSIONS:
                continue
            
            if spec.match_file(rel_path):
                continue
            
            allowed_files.append(rel_path)

    return allowed_files
