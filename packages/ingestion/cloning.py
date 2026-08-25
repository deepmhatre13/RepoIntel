import os
import shutil
import tempfile
import urllib.parse
from git import Repo

def validate_repo_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ["http", "https", "file"] or url.endswith(".git") or os.path.exists(url):
        return True
    return False

def clone_repository(repo_url: str, max_repo_size_mb: float = 500.0, clone_timeout_seconds: int = 180) -> str:
    if not validate_repo_url(repo_url):
        raise ValueError(f"Invalid repository URL or path: {repo_url}")
    
    target_dir = tempfile.mkdtemp(prefix="repointel_clone_")
    
    if os.path.isdir(repo_url):
        # Local repository directory copy for testing/evals
        shutil.copytree(repo_url, target_dir, dirs_exist_ok=True)
        return target_dir

    try:
        Repo.clone_from(repo_url, target_dir, depth=1)
    except Exception as e:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise RuntimeError(f"Git clone failed for {repo_url}: {str(e)}")
    
    # Calculate size
    total_bytes = 0
    for root, _, files in os.walk(target_dir):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    
    size_mb = total_bytes / (1024 * 1024)
    if size_mb > max_repo_size_mb:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise ValueError(f"Repository size ({size_mb:.2f} MB) exceeds maximum allowed ({max_repo_size_mb} MB)")
    
    return target_dir

def cleanup_repository(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
