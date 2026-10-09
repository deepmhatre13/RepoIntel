"""Repository Understanding Agent for Task A.

Uses Google Gemini via LLMClient to analyze real repository evidence (symbols,
files, dependencies, README chunks) and produce verified project overview,
technologies, architecture, key components, entry points, and evidence citations.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy import select

from packages.knowledge.db import AsyncSessionLocal
from packages.knowledge.models import (
    RepositoryModel, ArtifactModel, AgentRunModel, SymbolModel, DependencyModel, FileModel, CodeChunkModel
)
from packages.agentic.llm_client import LLMClient

logger = logging.getLogger(__name__)


async def generate_repository_understanding(
    repo_id: str,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Generate structured repository understanding backed by real evidence."""
    if llm_client is None:
        llm_client = LLMClient()

    async with AsyncSessionLocal() as session:
        repo = (
            await session.execute(
                select(RepositoryModel).where(RepositoryModel.id == repo_id)
            )
        ).scalar_one_or_none()

        if repo is None:
            return {"error": "Repository not found"}

        # Real symbols
        symbols_res = await session.execute(
            select(SymbolModel)
            .where(SymbolModel.repo_id == repo_id)
            .limit(30)
        )
        symbols = symbols_res.scalars().all()

        # Real dependencies
        deps_res = await session.execute(
            select(DependencyModel)
            .where(DependencyModel.repo_id == repo_id)
            .limit(20)
        )
        deps = deps_res.scalars().all()

        # Real files
        files_res = await session.execute(
            select(FileModel.file_path)
            .where(FileModel.repo_id == repo_id)
            .limit(40)
        )
        file_paths = files_res.scalars().all()

        # Retrieve any generated README
        artifact_res = await session.execute(
            select(ArtifactModel)
            .join(AgentRunModel, AgentRunModel.run_id == ArtifactModel.run_id)
            .where(AgentRunModel.repo_id == repo_id)
            .order_by(ArtifactModel.created_at.desc())
            .limit(1)
        )
        artifact = artifact_res.scalar_one_or_none()
        readme_text = artifact.content if artifact else ""

    # Build evidence summary
    evidence_list = []
    symbol_names = []
    for s in symbols[:15]:
        name = s.symbol_name or "unknown"
        path = s.file_path or ""
        start = s.line_start or 1
        end = s.line_end or 50
        symbol_names.append(name)
        if path:
            evidence_list.append({
                "file_path": path,
                "line_start": start,
                "line_end": end,
                "symbol_name": name,
                "snippet": f"Symbol '{name}' ({s.symbol_type}) in {path}",
            })

    dep_names = [d.package_name for d in deps if d.package_name]

    # Grounded fallback if Gemini is offline or degraded
    if not llm_client.is_available():
        return {
            "purpose": f"Repository {repo.repo_url} contains structured source code for {repo.archetype or 'software project'}.",
            "technologies": dep_names[:8] if dep_names else [repo.archetype or "Codebase"],
            "architecture": f"Organized into {len(file_paths)} files and {len(symbols)} AST symbols.",
            "key_components": [
                {"name": s.symbol_name, "file_path": s.file_path, "lines": f"{s.line_start}-{s.line_end}"}
                for s in symbols[:5]
            ],
            "entry_points": [
                {"name": f, "file_path": f, "lines": "1-100"}
                for f in file_paths if any(k in f for k in ["main", "app", "index", "cli", "run"])
            ][:4] or [{"name": file_paths[0] if file_paths else "root", "file_path": file_paths[0] if file_paths else "root", "lines": "1-50"}],
            "evidence": evidence_list[:8],
            "degraded": True,
        }

    # Format real evidence for Google Gemini prompt
    system_prompt = (
        "You are RepoIntel, an expert repository understanding agent using Google Gemini.\n"
        "Analyze the provided repository facts and evidence.\n"
        "Return ONLY a strict JSON object with these exact keys:\n"
        "{\n"
        '  "purpose": "High-level goal and purpose of repository",\n'
        '  "technologies": ["List of main frameworks and libraries supported by evidence"],\n'
        '  "architecture": "Clear overview of architecture based on code evidence",\n'
        '  "key_components": [{"name": "...", "description": "...", "file_path": "...", "lines": "..."}],\n'
        '  "entry_points": [{"name": "...", "file_path": "...", "lines": "..."}]\n'
        "}\n"
        "Do NOT invent files, symbols, or dependencies. If evidence is missing, state 'Insufficient evidence'."
    )

    user_context = (
        f"Repository URL: {repo.repo_url}\n"
        f"Archetype: {repo.archetype}\n"
        f"Discovered Files: {json.dumps(file_paths[:25])}\n"
        f"Dependencies: {json.dumps(dep_names[:15])}\n"
        f"AST Symbols: {json.dumps(symbol_names[:15])}\n"
        f"README Preview: {readme_text[:1000] if readme_text else 'None'}\n"
    )

    try:
        response_text = llm_client.generate(system_prompt, user_context)
        clean_json = response_text.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        parsed = json.loads(clean_json.strip())
        parsed["evidence"] = evidence_list[:8]
        parsed["degraded"] = False
        return parsed
    except Exception as exc:
        logger.warning("Gemini repository understanding fallback due to: %s", exc)
        return {
            "purpose": f"Repository analysis for {repo.repo_url}.",
            "technologies": dep_names[:6],
            "architecture": f"Repository structured with {len(file_paths)} indexed files and {len(symbols)} symbols.",
            "key_components": [
                {"name": s.symbol_name, "file_path": s.file_path, "lines": f"{s.line_start}-{s.line_end}"}
                for s in symbols[:4]
            ],
            "entry_points": [
                {"name": f, "file_path": f, "lines": "1-100"} for f in file_paths[:3]
            ],
            "evidence": evidence_list[:8],
            "degraded": True,
        }
