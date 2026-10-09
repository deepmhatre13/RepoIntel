import os
from typing import Tuple
from packages.shared import IngestionManifest
from packages.ingestion.cloning import clone_repository, cleanup_repository
from packages.ingestion.filtering import filter_repository
from packages.ingestion.metadata import extract_metadata, detect_archetype

def run_ingestion_pipeline(
    repo_url: str,
    max_repo_size_mb: float = 500.0,
    clone_timeout_seconds: int = 180,
    *,
    allowed_local_root: str | None = None,
) -> Tuple[IngestionManifest, list[str]]:
    temp_dir = clone_repository(
        repo_url,
        max_repo_size_mb,
        clone_timeout_seconds,
        allowed_local_root=allowed_local_root,
    )
    try:
        filtered_files = filter_repository(temp_dir)
        languages, manifest_files, entrypoints, ci_configs = extract_metadata(temp_dir, filtered_files)
        archetype = detect_archetype(temp_dir, filtered_files, manifest_files)

        total_bytes = 0
        for rel_f in filtered_files:
            fp = os.path.join(temp_dir, rel_f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)
        size_mb = total_bytes / (1024 * 1024)

        manifest = IngestionManifest(
            repo_url=repo_url,
            local_path=temp_dir,
            size_mb=size_mb,
            file_count=len(filtered_files),
            languages=languages,
            manifest_files=manifest_files,
            entrypoints=entrypoints,
            ci_configs=ci_configs,
            archetype=archetype
        )
        return manifest, filtered_files
    except Exception:
        cleanup_repository(temp_dir)
        raise
