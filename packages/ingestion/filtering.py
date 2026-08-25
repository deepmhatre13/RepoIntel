import os
import pathspec

IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".idea", ".vscode", ".next", "coverage", ".pytest_cache", ".mypy_cache"
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf", ".zip", ".tar",
    ".gz", ".7z", ".exe", ".dll", ".so", ".dylib", ".pyc", ".pyo", ".db",
    ".sqlite", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3"
}

def load_gitignore_spec(repo_path: str) -> pathspec.PathSpec:
    gitignore_path = os.path.join(repo_path, ".gitignore")
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
            patterns = f.readlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

def filter_repository(repo_path: str) -> list[str]:
    spec = load_gitignore_spec(repo_path)
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
