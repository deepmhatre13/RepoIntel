import hashlib
import re
from typing import Any, Dict, Iterable, List

from packages.agentic.state.schema import ArtifactSection


def _node_id(prefix: str, label: str) -> str:
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _label(value: Any) -> str:
    return re.sub(r"[\r\n\t]+", " ", str(value)).replace('"', "&quot;").strip()


def generate_mermaid_architecture(
    symbol_graph: Dict[str, Any],
    dependency_manifest: Iterable[Dict[str, Any]],
) -> ArtifactSection | None:
    modules = sorted({str(module).strip() for module in symbol_graph.get("modules", []) if str(module).strip()})
    if not modules:
        return None

    module_set = set(modules)
    lines = ["flowchart TD"]
    module_ids = {module: _node_id("module", module) for module in modules}
    for module in modules:
        lines.append(f'    {module_ids[module]}["{_label(module)}"]')

    edges = set()
    for edge in symbol_graph.get("module_edges", []):
        source = str(edge.get("source_component") or "").strip()
        target = str(edge.get("target_component") or "").strip()
        if source in module_set and target in module_set and source != target:
            edges.add((source, target))
    for source, target in sorted(edges):
        lines.append(f"    {module_ids[source]} --> {module_ids[target]}")

    dependencies = {}
    for dependency in dependency_manifest:
        package = str(dependency.get("package_name") or "").strip()
        if package:
            ecosystem = str(dependency.get("ecosystem") or "").strip()
            version = str(dependency.get("version_spec") or "").strip()
            detail = " ".join(value for value in (ecosystem, version) if value)
            dependencies.setdefault(package, detail)

    if dependencies:
        root_id = _node_id("repository", "Repository")
        lines.append(f'    {root_id}["Repository"]')
        for package, detail in sorted(dependencies.items()):
            package_id = _node_id("dependency", package)
            package_label = f"{package} ({detail})" if detail else package
            lines.append(f'    {package_id}["{_label(package_label)}"]')
            lines.append(f"    {root_id} -. declared dependency .-> {package_id}")

    content = "```mermaid\n" + "\n".join(lines) + "\n```"
    return ArtifactSection(
        section_title="Architecture Diagram",
        content=content,
        citations=[],
        insufficient_evidence=False,
    )


def extract_mermaid_node_labels(content: str) -> List[str]:
    return [
        match.group(1).replace("&quot;", '"')
        for match in re.finditer(r'^\s*[A-Za-z0-9_]+\["([^"\n]*)"\]', content, re.MULTILINE)
    ]