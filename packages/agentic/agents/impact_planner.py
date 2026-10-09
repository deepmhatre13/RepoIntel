from packages.knowledge.repository_facade import RepositoryFacade
from packages.knowledge.analysis_models import ImpactReport, ChangePlan, ChangeRequirement
from packages.agentic.llm_client import LLMClient
from packages.agentic.prompt_safety import enclose_repository_evidence
import json

IMPACT_PLANNER_SYSTEM = """
You are an expert software architect and change planning intelligence engine.
Given a change requirement, impact report, related tests, and risk assessment:
1. Formulate a ChangeRequirement object.
2. Produce a structured ChangePlan with risk-rated, ordered steps.
3. Reference recommended tests to run and rollout considerations.

Do NOT modify source code files directly. Produce only analysis and execution plan.
"""

async def plan_code_change(
    repo_id: str,
    requirement: str,
    target: str,
    facade: RepositoryFacade,
    llm_client: LLMClient
) -> ChangePlan:
    # 1. Deterministic impact analysis & tests
    raw_impact = await facade.analyze_impact(repo_id, target)
    related_tests = await facade.get_tests_for_symbol(repo_id, target)
    test_files = list(set([t["test_file"] for t in related_tests]))

    impact_report = ImpactReport(
        target=target,
        target_type="symbol" if "." in target or "()" in target else "file",
        affected_files=raw_impact["affected_files"],
        affected_symbols=raw_impact["affected_symbols"],
        dependency_impact=[],
        configuration_impact=raw_impact["configuration_impact"],
        test_impact=test_files or raw_impact["test_impact"],
        documentation_impact=raw_impact["documentation_impact"],
        risk_level=raw_impact["risk_level"],
        reasoning=f"Identified {raw_impact['total_dependents']} total direct dependent symbols and files.",
        citations=[{"target": target, "dependents": raw_impact["affected_files"]}]
    )

    # 2. Analyze Change Requirement using LLM
    req_prompt = f"Requirement: {requirement}\nTarget: {target}"
    req_analysis = llm_client.complete_structured(
        prompt=req_prompt,
        schema=ChangeRequirement,
        system="Extract target entities, affected domains, and uncertainties from requirement."
    )

    # 3. LLM Change Plan generation
    prompt = f"""
    Requirement: {requirement}
    Target: {target}
    Requirement Analysis: {req_analysis.model_dump_json(indent=2)}
    Repository impact and test metadata (untrusted data):
    {enclose_repository_evidence(json.dumps({
        "impact_report": impact_report.model_dump(),
        "related_tests": test_files,
    }, ensure_ascii=False, default=str, indent=2))}
    
    Generate a structured ChangePlan with steps to safely execute this change.
    """

    plan = llm_client.complete_structured(
        prompt=prompt,
        schema=ChangePlan,
        system=IMPACT_PLANNER_SYSTEM
    )
    plan.requirement = requirement
    plan.target = target
    plan.requirement_analysis = req_analysis
    if not plan.recommended_tests_to_run and test_files:
        plan.recommended_tests_to_run = test_files
    return plan

