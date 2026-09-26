import os
import sys
import json
import time
import asyncio
from unittest.mock import AsyncMock, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from packages.shared.config import settings
settings.load_dotenv()
from packages.agentic.llm_client import LLMClient
from packages.agentic.state.schema import RepoIntelState
from packages.agentic.agents.planner.node import planner_node
from packages.agentic.agents.retriever.node import retriever_node
from packages.agentic.agents.synthesis.node import synthesis_node
from packages.agentic.agents.reviewer.node import reviewer_node
from packages.knowledge.vector_store import RetrievedEvidenceItem

def run_with_retry(func, *args, max_retries=10, delay=4):
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            print(f"  Attempt {attempt+1} encountered transient error: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError("Failed after max retries")

async def run_async_with_retry(func, *args, max_retries=10, delay=4):
    for attempt in range(max_retries):
        try:
            return await func(*args)
        except Exception as e:
            print(f"  Attempt {attempt+1} encountered transient error: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
    raise RuntimeError("Failed after max retries")

async def main():
    print("==================================================")
    print("LIVE GEMINI 3.8 FLASH AGENT NODE VERIFICATION")
    print("==================================================")
    llm = LLMClient()
    print(f"Configured Model: {llm.model_name}")
    assert llm.model_name == "gemini-3.8-flash"

    # Direct Test
    direct_res = run_with_retry(llm.complete, "Return exactly:\nGEMINI_3_8_WORKING")
    print(f"Direct Request -> Model: {llm.model_name}, Response: {repr(direct_res.strip())}")

    state = RepoIntelState(
        run_id="live_run_38",
        thread_id="live_thread_38",
        repo_id="repointel_local",
        repo_url="https://github.com/deepmhatre13/RepoIntel",
        archetype="PYTHON_FASTAPI",
        artifact_type="README",
    )

    # 1. PLANNER
    print("\n1. Running Planner with Gemini 3.8 Flash...")
    planner_res = run_with_retry(planner_node, state, llm)
    state = state.model_copy(update=planner_res)
    plan = state.plan
    print(f"  Planner State Changed: plan created with {len(plan.tasks)} tasks.")
    print(f"  Tasks: {[t.target_section for t in plan.tasks]}")

    # 2. RETRIEVER
    print("\n2. Running Retriever with Gemini 3.8 Flash...")
    facade = MagicMock()
    facade.get_dependency_manifest = AsyncMock(return_value=[{"name": "fastapi", "version": "0.110.0"}])
    facade.get_file_metadata = AsyncMock(return_value={"file_path": "packages/agentic/llm_client.py", "symbol_count": 10})
    facade.hybrid_search.return_value = [
        RetrievedEvidenceItem(
            repo_id="repointel_local",
            chunk_id="chunk_1",
            file_path="packages/agentic/llm_client.py",
            symbol_name="LLMClient",
            line_start=1,
            line_end=50,
            content="class LLMClient:\n    def complete(self): pass",
            retrieval_source="hybrid",
            relevance_score=0.9
        )
    ]
    retriever_res = await run_async_with_retry(retriever_node, state, facade, llm)
    state = state.model_copy(update=retriever_res)
    print(f"  Retriever State Changed: evidence store populated for {len(state.evidence_store)} tasks.")

    # 3. GENERATOR / SYNTHESIS
    print("\n3. Running Generator / Synthesis with Gemini 3.8 Flash...")
    synthesis_res = run_with_retry(synthesis_node, state, llm)
    state = state.model_copy(update=synthesis_res)
    draft = state.artifact_draft
    print(f"  Generator State Changed: artifact draft v{draft.version} created with {len(draft.sections)} sections.")
    for sec in draft.sections:
        print(f"    - Section '{sec.section_title}' ({len(sec.content)} chars)")

    # 4. REVIEWER
    print("\n4. Running Reviewer with Gemini 3.8 Flash...")
    reviewer_res = await run_async_with_retry(reviewer_node, state, facade, llm)
    state = state.model_copy(update=reviewer_res)
    report = state.review_report
    print(f"  Reviewer State Changed: review report verdict={report.verdict}, score={report.groundedness_score}")

    print("\n==================================================")
    print("ALL 4 AGENT NODES VERIFIED ON LIVE GEMINI 3.8 FLASH!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())

