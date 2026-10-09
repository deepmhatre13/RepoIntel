from packages.agentic.agents.synthesis.quickstart import generate_grounded_quickstart
from packages.knowledge.repository_facade import RepositoryFacade
from packages.knowledge.vector_store import VectorStore


def test_quickstart_emits_only_detected_facts_with_source_citations():
    facts = {
        "file_paths": [
            "src/main.py",
            "package-lock.json",
            ".github/workflows/ci.yml",
            "LICENSE",
        ],
        "entrypoints": ["src/main.py"],
        "files": {
            "src/main.py": [{
                "line_start": 1,
                "line_end": 5,
                "content": "def main():\n    start_server()",
            }],
            "package-lock.json": [{
                "line_start": 1,
                "line_end": 2,
                "content": '{"name":"demo","lockfileVersion":3}',
            }],
            ".github/workflows/ci.yml": [{
                "line_start": 4,
                "line_end": 8,
                "content": "steps:\n  - run: npm test",
            }],
            "LICENSE": [{
                "line_start": 1,
                "line_end": 2,
                "content": "MIT License\nPermission is hereby granted.",
            }],
        },
    }

    section = generate_grounded_quickstart(facts)

    assert section is not None
    assert "`src/main.py`" in section.content
    assert "`npm`" in section.content
    assert "`npm test`" in section.content
    assert "License: MIT" in section.content
    assert {citation.file_path for citation in section.citations} == {
        "src/main.py",
        "package-lock.json",
        ".github/workflows/ci.yml",
        "LICENSE",
    }


def test_quickstart_omits_undetected_commands_and_license_badges():
    section = generate_grounded_quickstart({
        "file_paths": ["src/app.py", "LICENSE"],
        "entrypoints": [],
        "files": {
            "src/app.py": [{
                "line_start": 1,
                "line_end": 1,
                "content": "def app(): pass",
            }],
            "LICENSE": [{
                "line_start": 1,
                "line_end": 1,
                "content": "License terms are available from the authors.",
            }],
        },
    })

    assert section is None
    assert generate_grounded_quickstart({"file_paths": [], "files": {}}) is None


async def test_facade_returns_indexed_quickstart_source_chunks_only():
    class Result:
        def __init__(self, values):
            self.values = values

        def scalars(self):
            return self

        def all(self):
            return self.values

    class Session:
        def __init__(self):
            self.query_count = 0

        async def execute(self, statement):
            self.query_count += 1
            if self.query_count == 1:
                return Result(["src/main.py", "README.md"])
            return Result([type("Chunk", (), {
                "file_path": "src/main.py",
                "line_start": 1,
                "line_end": 3,
                "content": "def main(): pass",
            })()])

    facade = RepositoryFacade(Session(), VectorStore())

    async def manifest(repo_id):
        return {
            "manifest_data": {
                "entrypoints": ["src/main.py"],
                "manifest_files": [],
            }
        }

    facade.get_ingestion_manifest = manifest
    facts = await facade.get_quickstart_facts("repo-1")

    assert facts["entrypoints"] == ["src/main.py"]
    assert set(facts["files"]) == {"src/main.py"}
    assert facts["files"]["src/main.py"][0]["line_start"] == 1