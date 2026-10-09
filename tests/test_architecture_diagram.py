from packages.agentic.agents.synthesis.diagram import (
    extract_mermaid_node_labels,
    generate_mermaid_architecture,
)
from packages.agentic.agents.reviewer.node import reviewer_node
from packages.agentic.state.schema import (
    ArtifactDraft,
    RepoIntelState,
    ReviewReport,
    ReviewVerdict,
)


def test_architecture_diagram_uses_only_real_modules_edges_and_dependencies():
    section = generate_mermaid_architecture(
        {
            "modules": ["api", "knowledge", "worker"],
            "module_edges": [
                {"source_component": "api", "target_component": "knowledge"},
                {"source_component": "worker", "target_component": "knowledge"},
                {"source_component": "invented", "target_component": "api"},
            ],
        },
        [
            {"package_name": "fastapi", "ecosystem": "pypi", "version_spec": ">=0.1"},
            {"package_name": "", "ecosystem": "pypi", "version_spec": "*"},
        ],
    )

    assert section is not None
    labels = set(extract_mermaid_node_labels(section.content))
    assert labels == {"api", "knowledge", "worker", "Repository", "fastapi (pypi >=0.1)"}
    assert " --> " in section.content
    assert "invented" not in section.content
    assert section.content.count("declared dependency") == 1


def test_architecture_diagram_escapes_quotes_and_omits_empty_graph():
    section = generate_mermaid_architecture(
        {"modules": ['api"unsafe'], "module_edges": []}, []
    )

    assert section is not None
    assert "api&quot;unsafe" in section.content
    assert extract_mermaid_node_labels(section.content) == ['api"unsafe']
    assert generate_mermaid_architecture({"modules": []}, []) is None


class ArchitectureFacade:
    async def get_architecture_snapshot(self, repo_id):
        return {
            "modules": ["api", "knowledge"],
            "module_edges": [{
                "source_component": "api",
                "target_component": "knowledge",
            }],
        }

    async def get_dependency_manifest(self, repo_id):
        return [{"package_name": "fastapi", "ecosystem": "pypi", "version_spec": "*"}]


class ApprovingReviewer:
    def complete_structured(self, prompt, schema, system=None):
        return ReviewReport(verdict=ReviewVerdict.APPROVE)


def _review_state(section):
    return RepoIntelState(
        run_id="run-1",
        thread_id="thread-1",
        repo_id="repo-1",
        repo_url="https://example.com/repo",
        archetype="SERVICE",
        artifact_draft=ArtifactDraft(version=1, sections=[section]),
    )


async def test_reviewer_rejects_diagram_with_ungrounded_module():
    expected = generate_mermaid_architecture(
        await ArchitectureFacade().get_architecture_snapshot("repo-1"),
        await ArchitectureFacade().get_dependency_manifest("repo-1"),
    )
    tampered = expected.model_copy(
        update={"content": expected.content.replace('"api"', '"invented"')}
    )

    result = await reviewer_node(
        _review_state(tampered), ArchitectureFacade(), ApprovingReviewer()
    )

    assert result["review_report"].verdict == ReviewVerdict.REVISE_SYNTHESIS
    assert any(
        "diagram" in issue.description.lower()
        for issue in result["review_report"].issues
    )