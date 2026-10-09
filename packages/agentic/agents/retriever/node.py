import json
from typing import Dict, Any
from packages.agentic.state.schema import RepoIntelState, TaskStatus
from packages.agentic.llm_client import LLMClient
from packages.knowledge.repository_facade import RepositoryFacade
from packages.agentic.core.tool_loop import ToolCallDecision, run_tool_selection_loop

MAX_TOOL_ROUNDS = 4

async def retriever_node(state: RepoIntelState, facade: RepositoryFacade, llm_client: LLMClient) -> Dict[str, Any]:
    if not state.plan or not state.plan.tasks:
        return {"errors": state.errors + ["Retriever executed with empty plan"]}

    evidence_store = dict(state.evidence_store)

    for task in state.plan.tasks:
        loop_result = await run_tool_selection_loop(
            llm_client,
            facade,
            task.objective,
            MAX_TOOL_ROUNDS,
            repo_id=state.repo_id,
            query=task.retrieval_query,
            evidence_type=task.evidence_type,
            target_section=task.target_section,
            task_id=task.task_id,
        )
        task.status = TaskStatus.COMPLETED
        evidence_store[task.task_id] = loop_result.evidence

    return {"evidence_store": evidence_store, "plan": state.plan}
