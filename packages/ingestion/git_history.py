import os
import json
import subprocess
from datetime import datetime, timezone
from typing import List, Dict, Any
from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from packages.knowledge.models import (
    CommitModel, FileChangeModel, SymbolHistoryModel, SymbolChangeModel,
    CoChangeRelationModel, APIChangeModel, DependencyChangeModel,
    CommitAnalysisModel
)
from packages.knowledge.diff_analysis import DiffAnalyzer

def parse_git_commits(repo_path: str, max_depth: int = 50) -> List[Dict[str, Any]]:
    if not os.path.exists(os.path.join(repo_path, ".git")):
        return []
    try:
        cmd = ["git", "log", f"-n{max_depth}", "--pretty=format:%H|%P|%an|%at|%s"]
        # encoding="utf-8" + errors="replace": git output is UTF-8; on Windows the
        # default locale codec (cp1252) crashes on non-ASCII author names /
        # commit subjects, which previously caused silent total history loss
        # (the outer except returned []). Phase T: measured on the real corpus.
        res = subprocess.run(
            cmd, cwd=repo_path, capture_output=True, text=True, check=True,
            encoding="utf-8", errors="replace",
        )
        lines = res.stdout.strip().split("\n")
        commits = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split("|", 4)
            if len(parts) < 5:
                continue
            sha, parent_sha, author, ts_str, msg = parts
            dt = datetime.fromtimestamp(int(ts_str), tz=timezone.utc) if ts_str.isdigit() else datetime.now(timezone.utc)

            diff_cmd = ["git", "show", "--numstat", "--pretty=format:", sha]
            diff_res = subprocess.run(
                diff_cmd, cwd=repo_path, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            changes = []
            for dline in diff_res.stdout.strip().split("\n"):
                if not dline.strip():
                    continue
                dparts = dline.split("\t")
                if len(dparts) >= 3:
                    adds = int(dparts[0]) if dparts[0].isdigit() else 0
                    dels = int(dparts[1]) if dparts[1].isdigit() else 0
                    filepath = dparts[2]
                    changes.append({
                        "file_path": filepath,
                        "change_type": "MODIFIED",
                        "additions": adds,
                        "deletions": dels
                    })
            commits.append({
                "sha": sha,
                "parent_sha": parent_sha.split()[0] if parent_sha else None,
                "author": author,
                "message": msg,
                "timestamp": dt,
                "file_changes": changes
            })
        return commits
    except Exception:
        return []


async def persist_git_commits(
    session: AsyncSession,
    repo_id: str,
    commits: List[Dict[str, Any]],
    parsed_symbols=None,
    repo_path: str = None,
    max_history_commits: int = 50,
    max_diff_size: int = 100000,
    max_co_change_pairs: int = 2000,
):
    """Persist git commits with AST-aware diff analysis (Phase 23).

    When repo_path is provided, uses DiffAnalyzer to:
    - Map changed line ranges to specific symbols (not whole files)
    - Classify changes (SIGNATURE_CHANGE, BODY_CHANGE, etc.)
    - Detect API changes and dependency changes
    - Store CommitAnalysis for each commit
    - Aggregate CoChange relations (no duplicates)
    """
    if repo_path is None:
        # Fallback: legacy behavior without diff analysis
        repo_path = ""

    # Map parsed symbols by file_path
    symbols_by_file = {}
    if parsed_symbols:
        for s in parsed_symbols:
            symbols_by_file.setdefault(getattr(s, 'file_path', ''), []).append(s)

    diff_analyzer = DiffAnalyzer(max_diff_size=max_diff_size) if repo_path else None

    # Idempotency (Phase 40.4 / 43.4): skip commits already persisted for this repo
    # so re-running analysis never duplicates commits or their derived rows.
    existing_shas = set()
    if commits:
        existing_res = await session.execute(
            select(CommitModel.sha).where(
                CommitModel.repo_id == repo_id,
                CommitModel.sha.in_([c["sha"] for c in commits[:max_history_commits]]),
            )
        )
        existing_shas = {sha for (sha,) in existing_res.all()}

    for c in commits[:max_history_commits]:
        if c["sha"] in existing_shas:
            continue
        commit_obj = CommitModel(
            repo_id=repo_id,
            sha=c["sha"],
            parent_sha=c.get("parent_sha"),
            author=c["author"],
            message=c["message"],
            timestamp=c.get("timestamp", datetime.now(timezone.utc))
        )
        session.add(commit_obj)

        changed_symbols_in_commit = []
        changed_files_list = []
        changed_symbols_list = []
        api_changes_list = []
        dependency_changes_list = []
        configuration_changes_list = []
        risk_signals = []

        # Use AST-aware diff analysis when repo_path is available
        diff_analysis = None
        if diff_analyzer and repo_path and c.get("parent_sha"):
            try:
                diff_analysis = diff_analyzer.analyze_commit(
                    repo_path=repo_path,
                    commit_sha=c["sha"],
                    parent_sha=c["parent_sha"],
                    parsed_symbols=parsed_symbols or []
                )
            except Exception:
                diff_analysis = None

        if diff_analysis:
            changed_files_list = diff_analysis.get("changed_files", [])
            changed_symbols_list = diff_analysis.get("changed_symbols", [])
            api_changes_list = diff_analysis.get("api_changes", [])
            dependency_changes_list = diff_analysis.get("dependency_changes", [])
            configuration_changes_list = diff_analysis.get("configuration_changes", [])
            risk_signals = diff_analysis.get("risk_signals", [])

        # Store FileChangeModel entries
        for fc in c.get("file_changes", []):
            fc_obj = FileChangeModel(
                commit_sha=c["sha"],
                repo_id=repo_id,
                file_path=fc["file_path"],
                change_type=fc["change_type"],
                additions=fc["additions"],
                deletions=fc["deletions"]
            )
            session.add(fc_obj)

        # Store SymbolHistory and SymbolChange entries
        if diff_analysis:
            # Use diff-aware symbol mapping
            for sym_info in changed_symbols_list:
                sym_name = sym_info["symbol_name"]
                fpath = sym_info["file_path"]
                line_start = sym_info.get("line_start")
                line_end = sym_info.get("line_end")

                # Compute content hash from current symbol
                content_hash = None
                file_symbols = symbols_by_file.get(fpath, [])
                for s in file_symbols:
                    if getattr(s, 'symbol_name', None) == sym_name or getattr(s, 'name', None) == sym_name:
                        break

                # Build change summary
                change_summary = f"{sym_info.get('change_category', 'BODY_CHANGE')}: lines {line_start}-{line_end}"
                evidence = json.dumps({"hunk_line_range": [line_start, line_end], "confidence": sym_info.get("confidence", "HIGH")})

                sh_obj = SymbolHistoryModel(
                    repo_id=repo_id,
                    symbol_name=sym_name,
                    file_path=fpath,
                    commit_sha=c["sha"],
                    author=c["author"],
                    change_type=fc["change_type"] if c.get("file_changes") else "MODIFIED",
                    timestamp=c.get("timestamp", datetime.now(timezone.utc)),
                    line_start=line_start,
                    line_end=line_end,
                    content_hash=content_hash,
                    change_summary=change_summary,
                    evidence=evidence
                )
                session.add(sh_obj)

                sc_obj = SymbolChangeModel(
                    repo_id=repo_id,
                    commit_sha=c["sha"],
                    file_path=fpath,
                    symbol_name=sym_name,
                    change_category=sym_info.get("change_category", "BODY_CHANGE"),
                    line_start=line_start,
                    line_end=line_end,
                    before_signature=None,
                    after_signature=sym_info.get("signature"),
                    confidence=sym_info.get("confidence", "HIGH"),
                    evidence=evidence,
                    source_index=len(changed_symbols_in_commit)
                )
                session.add(sc_obj)
                changed_symbols_in_commit.append(sym_name)
        else:
            # Fallback: mark symbols in changed files
            for fc in c.get("file_changes", []):
                file_symbols = symbols_by_file.get(fc["file_path"], [])
                for s in file_symbols:
                    sname = getattr(s, 'name', getattr(s, 'symbol_name', str(s)))
                    sh_obj = SymbolHistoryModel(
                        repo_id=repo_id,
                        symbol_name=sname,
                        file_path=fc["file_path"],
                        commit_sha=c["sha"],
                        author=c["author"],
                        change_type=fc["change_type"],
                        timestamp=c.get("timestamp", datetime.now(timezone.utc))
                    )
                    session.add(sh_obj)

                    sc_obj = SymbolChangeModel(
                        repo_id=repo_id,
                        commit_sha=c["sha"],
                        file_path=fc["file_path"],
                        symbol_name=sname,
                        change_category="BODY_CHANGE"
                    )
                    session.add(sc_obj)
                    changed_symbols_in_commit.append(sname)

        # Store API changes
        for ac in api_changes_list:
            ac_obj = APIChangeModel(
                repo_id=repo_id,
                commit_sha=c["sha"],
                api_symbol=ac["api_symbol"],
                before_signature=ac.get("before_signature"),
                after_signature=ac.get("after_signature"),
                change_type=ac.get("change_type", "MODIFIED"),
                confidence=ac.get("confidence", "HIGH"),
                evidence=ac.get("evidence", "")
            )
            session.add(ac_obj)

        # Store dependency changes
        for dc in dependency_changes_list:
            dc_obj = DependencyChangeModel(
                repo_id=repo_id,
                commit_sha=c["sha"],
                package_name=dc["package_name"],
                ecosystem=dc["ecosystem"],
                change_type=dc["change_type"],
                old_version=dc.get("old_version"),
                new_version=dc.get("new_version"),
                confidence=dc.get("confidence", "HIGH"),
                evidence=dc.get("evidence", "")
            )
            session.add(dc_obj)

        # Aggregate Co-Change relationships (set-based, idempotent).
        #
        # The original implementation issued one async SELECT per symbol pair
        # per commit. A shallow-clone root commit marks every parsed symbol as
        # changed (~2k symbols), so the O(k^2) pair loop executed ~2M database
        # round-trips inside a single uncommitted transaction -- the stage
        # appeared frozen for hours while holding the write lock.
        #
        # Now: pairs are aggregated in memory (dict keyed by the canonical
        # ordered pair), existing rows for this repo are fetched exactly once,
        # counters are incremented in memory, and the diff is written as bulk
        # adds. Uniqueness of (repo_id, symbol_a, symbol_b) is enforced by the
        # schema (migration 0014) so a concurrent/retried run cannot duplicate
        # a pair; ON CONFLICT is not needed because the in-memory map already
        # merged counts and the read happened inside the same transaction.
        unique_syms = list(dict.fromkeys(changed_symbols_in_commit))  # dedupe, order-stable
        pair_counts: Dict[tuple, Dict[str, Any]] = {}
        for i in range(len(unique_syms)):
            for j in range(i + 1, len(unique_syms)):
                if len(pair_counts) >= max_co_change_pairs:
                    break
                sa, sb = sorted([unique_syms[i], unique_syms[j]])
                key = (sa, sb)
                entry = pair_counts.get(key)
                if entry is None:
                    pair_counts[key] = {
                        "symbol_a": sa,
                        "symbol_b": sb,
                        "co_change_count": 1,
                        "supporting_commits": [c["sha"]],
                    }
                else:
                    entry["co_change_count"] += 1
                    entry["supporting_commits"].append(c["sha"])
            if len(pair_counts) >= max_co_change_pairs:
                break

        if pair_counts:
            keys = list(pair_counts.keys())
            existing_pairs = {}
            if keys:
                existing_res = await session.execute(
                    select(CoChangeRelationModel).where(
                        CoChangeRelationModel.repo_id == repo_id,
                        tuple_(CoChangeRelationModel.symbol_a, CoChangeRelationModel.symbol_b)
                        .in_([list(k) for k in keys]),
                    )
                )
                for row in existing_res.scalars():
                    existing_pairs[(row.symbol_a, row.symbol_b)] = row

            for key, agg in pair_counts.items():
                row = existing_pairs.get(key)
                if row is not None:
                    row.co_change_count = (row.co_change_count or 0) + agg["co_change_count"]
                    commits_json = json.loads(row.supporting_commits or "[]")
                    commits_json.extend(agg["supporting_commits"])
                    row.supporting_commits = json.dumps(commits_json[-200:])
                    if c.get("total_commits"):
                        row.co_change_ratio = round(
                            row.co_change_count / c["total_commits"], 4
                        )
                    session.add(row)
                else:
                    session.add(CoChangeRelationModel(
                        repo_id=repo_id,
                        symbol_a=agg["symbol_a"],
                        symbol_b=agg["symbol_b"],
                        co_change_count=agg["co_change_count"],
                        supporting_commits=json.dumps(agg["supporting_commits"][-200:]),
                    ))

        # Store CommitAnalysis
        ca_obj = CommitAnalysisModel(
            repo_id=repo_id,
            commit_sha=c["sha"],
            changed_files_json=json.dumps(changed_files_list),
            changed_symbols_json=json.dumps(
                [{"symbol_name": s, "change_category": sc.get("change_category", "BODY_CHANGE")}
                 for sc, s in zip(changed_symbols_list, changed_symbols_in_commit)]
            ),
            dependency_changes_json=json.dumps(dependency_changes_list),
            api_changes_json=json.dumps(api_changes_list),
            configuration_changes_json=json.dumps(configuration_changes_list),
            summary=f"Commit {c['sha'][:8]} by {c['author']}: {c['message'][:100]}",
            risk_signals_json=json.dumps(risk_signals)
        )
        session.add(ca_obj)

    await session.commit()
