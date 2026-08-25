import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from packages.agentic.state.schema import RepoIntelState, TaskEvidence, EvidenceCitation, TaskStatus
from packages.agentic.llm_client import LLMClient
from packages.knowledge.repository_facade import RepositoryFacade

class ToolCallDecision(BaseModel):
    tool_name: str = Field(description="Name of tool: hybrid_retrieve, symbol_graph_neighbors, get_dependency_manifest, get_file_metadata, finish_task")
    query_or_param: str = Field(description="Search query, file path, symbol name, or empty string")
    reasoning: str = Field(description="Reason for calling this tool")

RETRIEVER_SYSTEM_PROMPT = """
You are an expert repository retriever.
Your task is to select and execute tools to gather complete, factual code evidence for a specific documentation task.
Do NOT invent file paths, line numbers, or code contents.
All evidence MUST come from actual tool execution results.

Available Tools:
1. `hybrid_retrieve(query)`: Search for code chunks matching semantic query.
2. `symbol_graph_neighbors(symbol_or_file)`: Find caller/callee or import relationships.
3. `get_dependency_manifest()`: List dependencies extracted from manifest files.
4. `get_file_metadata(file_path)`: Get metadata and symbol list for a specific file.
5. `finish_task()`: Conclude retrieval when sufficient evidence has been gathered or options exhausted.

Examine the task and available history, then output a ToolCallDecision JSON.
"""

MAX_TOOL_ROUNDS = 4

async def retriever_node(state: RepoIntelState, facade: RepositoryFacade, llm_client: LLMClient) -> Dict[str, Any]:
    if not state.plan or not state.plan.tasks:
        return {"errors": state.errors + ["Retriever executed with empty plan"]}

    evidence_store = dict(state.evidence_store)

    for task in state.plan.tasks:
        citations: List[EvidenceCitation] = []
        reasoning_trail = []

        for round_idx in range(MAX_TOOL_ROUNDS):
            prompt = f"""
            Task Objective: {task.objective}
            Retrieval Query: {task.retrieval_query}
            Target Section: {task.target_section}
            Evidence Type: {task.evidence_type}
            
            Current Round: {round_idx + 1}/{MAX_TOOL_ROUNDS}
            Citations Gathered So Far: {len(citations)}
            Reasoning Trail: {json.dumps(reasoning_trail)}
            
            What tool should be called next?
            """

            decision = llm_client.complete_structured(
                prompt=prompt,
                schema=ToolCallDecision,
                system=RETRIEVER_SYSTEM_PROMPT
            )

            reasoning_trail.append(f"Round {round_idx + 1}: {decision.tool_name} -> {decision.reasoning}")

            if decision.tool_name == "finish_task":
                break
            elif decision.tool_name == "get_dependency_manifest":
                deps = await facade.get_dependency_manifest(state.repo_id)
                dep_content = json.dumps(deps, indent=2)
                citations.append(EvidenceCitation(
                    file_path="manifests/dependencies.json",
                    line_start=1,
                    line_end=max(1, len(deps)),
                    content=dep_content,
                    relevance_score=1.0
                ))
            elif decision.tool_name == "get_file_metadata":
                meta = await facade.get_file_metadata(state.repo_id, decision.query_or_param)
                if meta:
                    citations.append(EvidenceCitation(
                        file_path=decision.query_or_param,
                        line_start=1,
                        line_end=1,
                        content=json.dumps(meta),
                        relevance_score=0.9
                    ))
            elif decision.tool_name == "symbol_graph_neighbors":
                neighbors = await facade.symbol_graph_neighbors(state.repo_id, decision.query_or_param)
                if neighbors:
                    citations.append(EvidenceCitation(
                        file_path=decision.query_or_param,
                        line_start=1,
                        line_end=1,
                        content=json.dumps(neighbors),
                        relevance_score=0.85
                    ))
            else: # hybrid_retrieve
                query = decision.query_or_param or task.retrieval_query
                items = facade.hybrid_search(state.repo_id, query, top_k=5)
                for item in items:
                    citations.append(EvidenceCitation(
                        file_path=item.file_path,
                        line_start=item.line_start,
                        line_end=item.line_end,
                        symbol_name=item.symbol_name,
                        content=item.content,
                        relevance_score=item.relevance_score
                    ))

        task.status = TaskStatus.COMPLETED
        evidence_store[task.task_id] = TaskEvidence(
            task_id=task.task_id,
            query=task.retrieval_query,
            citations=citations,
            reasoning="\n".join(reasoning_trail)
        )

    return {"evidence_store": evidence_store, "plan": state.plan}
