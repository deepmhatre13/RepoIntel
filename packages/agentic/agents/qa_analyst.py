import json
from typing import Dict, Any, List
from packages.knowledge.repository_facade import RepositoryFacade
from packages.knowledge.analysis_models import QAAnswer
from packages.agentic.llm_client import LLMClient

PLANNER_QA_SYSTEM = """
You are a software repository analyst query planner.
Given a user question, decide which intelligence sources are required to provide a grounded, evidence-backed answer.

Output a JSON dict with keys:
- needs_semantic: bool
- needs_symbols: bool
- needs_graph: bool
- needs_call_chain: bool
- needs_git_history: bool
- needs_impact: bool
- needs_cross_repo: bool
- sub_queries: list of str
"""

QA_ANSWER_SYSTEM = """
You are RepoIntel, an expert repository analyst AI.
Answer the user query based ONLY on the repository evidence provided.
If evidence is insufficient, set confidence to "LOW" and state clearly:
"The repository does not provide enough evidence to determine this."

Do NOT hallucinate or invent code, symbols, files, or commit history.
"""

async def answer_repository_query(
    repo_id: str,
    query: str,
    facade: RepositoryFacade,
    llm_client: LLMClient,
    group_repo_ids: List[str] = None
) -> QAAnswer:
    # 1. Plan query retrieval
    plan_raw = llm_client.complete(
        f"Query: {query}",
        system=PLANNER_QA_SYSTEM
    )
    
    # 2. Gather evidence based on plan
    evidence_items = []
    related_files = set()
    related_symbols = set()

    # Hybrid Search (Semantic + Lexical)
    if group_repo_ids:
        retrieved = facade.multi_repository_hybrid_search(group_repo_ids, query, top_k=5)
    else:
        retrieved = facade.hybrid_search(repo_id, query, top_k=5)
        
    for item in retrieved:
        evidence_items.append({
            "repo_id": item.repo_id,
            "file_path": item.file_path,
            "line_start": item.line_start,
            "line_end": item.line_end,
            "symbol_name": item.symbol_name,
            "content": item.content
        })
        related_files.add(item.file_path)
        if item.symbol_name:
            related_symbols.add(item.symbol_name)

    # Git Commits if query mentions history/changes/commits
    related_commits = []
    if any(k in query.lower() for k in ["commit", "history", "changed", "recent", "who", "when", "evolved"]):
        commits = await facade.get_git_commits(repo_id, limit=5)
        for c in commits:
            related_commits.append(c['sha'])
            evidence_items.append({
                "source": "git_history",
                "content": f"Commit {c['sha'][:8]} by {c['author']}: {c['message']} (Files: {', '.join(c['changed_files'])})"
            })

    # Symbol History / Co-Change
    words = query.split()
    for w in words:
        if len(w) > 3:
            sh = await facade.get_symbol_history(repo_id, w)
            if sh:
                evidence_items.append({
                    "source": "symbol_history",
                    "content": f"Symbol '{w}' changed in commits: {[item['commit_sha'][:8] for item in sh[:3]]}"
                })

    # Test Mapping
    if any(k in query.lower() for k in ["test", "tests", "coverage"]):
        for w in words:
            tests = await facade.get_tests_for_symbol(repo_id, w)
            if tests:
                evidence_items.append({
                    "source": "test_to_code",
                    "content": f"Symbol '{w}' related to tests: {[t['test_name'] for t in tests]}"
                })

    # Call chain / Impact if query asks about flow or change impact
    if any(k in query.lower() for k in ["impact", "affect", "happen if", "trace", "flow"]):
        impact = await facade.analyze_impact(repo_id, query.split()[-1])
        evidence_items.append({
            "source": "impact_analysis",
            "content": f"Target '{impact['target']}' has risk level {impact['risk_level']}. Direct affected files: {impact['affected_files']}"
        })

    if not evidence_items:
        return QAAnswer(
            query=query,
            answer="The repository does not provide enough evidence to determine this.",
            confidence="LOW",
            evidence=[],
            related_files=[],
            related_symbols=[]
        )

    # 3. Synthesize answer with LLM
    context_str = json.dumps(evidence_items, indent=2)
    answer_text = llm_client.complete(
        f"User Query: {query}\n\nRepository Evidence:\n{context_str}",
        system=QA_ANSWER_SYSTEM
    )

    confidence = "HIGH" if len(evidence_items) >= 2 else "MEDIUM"
    if "does not provide enough evidence" in answer_text.lower():
        confidence = "LOW"

    return QAAnswer(
        query=query,
        answer=answer_text,
        confidence=confidence,
        evidence=evidence_items,
        related_files=list(related_files),
        related_symbols=list(related_symbols)
    )
