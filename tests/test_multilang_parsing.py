"""Tests for the multi-language structural parsing layer (Stage 2)."""

import pytest

from packages.parsing.normalized_model import (
    NormalizedSymbol,
    NormalizedEdge,
    NormalizedFile,
    CLASS,
    FUNCTION,
    METHOD,
    VARIABLE,
    IMPORT,
    STRUCT,
    ENUM,
    TRAIT,
    INTERFACE,
    TYPE_ALIAS,
    CALLS,
    INHERITS,
    IMPLEMENTS,
    CONTAINS,
    FROM_TREE_SITTER,
    FROM_PYTHON_AST,
    FROM_SCIP,
)
from packages.parsing.tree_sitter_parser import (
    parse_file_with_tree_sitter,
    is_language_supported,
    supported_extensions,
)
from packages.parsing.repository_parser import parse_file
from packages.parsing.scip_adapter import (
    get_scip_availability,
    is_scip_available,
    find_scip_index,
    SCIPAvailability,
)


class TestNormalizedModel:
    def test_normalized_symbol_creation(self):
        sym = NormalizedSymbol(
            repository_id="repo1",
            file_path="src/auth.py",
            language="Python",
            symbol_name="Authenticator",
            symbol_type=CLASS,
            line_start=1,
            line_end=10,
            signature="class Authenticator",
        )
        assert sym.symbol_name == "Authenticator"
        assert sym.symbol_type == CLASS
        assert sym.extraction_method == FROM_TREE_SITTER
        assert sym.confidence == 1.0

    def test_normalized_edge_creation(self):
        edge = NormalizedEdge(
            repository_id="repo1",
            source_symbol="Dog",
            source_file="src/models.py",
            edge_type=INHERITS,
            target_symbol="Animal",
            line_start=5,
            line_end=5,
        )
        assert edge.edge_type == INHERITS
        assert edge.source_symbol == "Dog"
        assert edge.target_symbol == "Animal"

    def test_normalized_file_creation(self):
        nf = NormalizedFile(
            repository_id="repo1",
            file_path="src/main.py",
            language="Python",
            parse_succeeded=True,
        )
        assert nf.parse_succeeded is True
        assert nf.symbols == []
        assert nf.classes == 0


class TestTreeSitterParser:
    def test_supported_extensions(self):
        exts = supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".ts" in exts
        assert ".java" in exts
        assert ".go" in exts
        assert ".rs" in exts
        assert ".c" in exts
        assert ".cpp" in exts

    def test_is_language_supported(self):
        assert is_language_supported(".py") is True
        assert is_language_supported(".js") is True
        assert is_language_supported(".ts") is True
        assert is_language_supported(".xyz") is False

    def test_parse_javascript(self):
        code = "function greet(name) { return 'Hello ' + name; }\nclass Animal { speak() { return 'noise'; } }\n"
        result = parse_file_with_tree_sitter("repo1", "test.js", code, ".js")
        assert result.parse_succeeded is True
        assert result.language == "JavaScript"
        assert len(result.symbols) >= 2
        names = [s.symbol_name for s in result.symbols]
        assert "greet" in names
        assert "Animal" in names

    def test_parse_typescript(self):
        code = "interface Identifiable { id: number; }\nclass User implements Identifiable { id: number = 0; }\n"
        result = parse_file_with_tree_sitter("repo1", "test.ts", code, ".ts")
        assert result.parse_succeeded is True
        assert result.language == "TypeScript"
        assert len(result.symbols) >= 1

    def test_parse_go(self):
        code = "package main\ntype Authenticator interface { Authenticate() bool }\nfunc main() {}\n"
        result = parse_file_with_tree_sitter("repo1", "main.go", code, ".go")
        assert result.parse_succeeded is True
        assert result.language == "Go"
        assert len(result.symbols) >= 1

    def test_parse_rust(self):
        code = "trait Authenticator { fn authenticate(&self) -> bool; }\nstruct JWTValidator;\nimpl Authenticator for JWTValidator { fn authenticate(&self) -> bool { true } }\n"
        result = parse_file_with_tree_sitter("repo1", "auth.rs", code, ".rs")
        assert result.parse_succeeded is True
        assert result.language == "Rust"
        names = [s.symbol_name for s in result.symbols]
        assert "Authenticator" in names
        assert "JWTValidator" in names

    def test_parse_java(self):
        code = "interface Authenticator { boolean authenticate(); }\nclass JWTValidator implements Authenticator { public boolean authenticate() { return true; } }\n"
        result = parse_file_with_tree_sitter("repo1", "Auth.java", code, ".java")
        assert result.parse_succeeded is True
        assert result.language == "Java"
        names = [s.symbol_name for s in result.symbols]
        assert "Authenticator" in names
        assert "JWTValidator" in names

    def test_parse_c(self):
        code = "struct Point { int x; int y; };\nint add(int a, int b) { return a + b; }\n"
        result = parse_file_with_tree_sitter("repo1", "math.c", code, ".c")
        assert result.parse_succeeded is True
        assert result.language == "C"
        assert len(result.symbols) >= 1

    def test_parse_cpp(self):
        code = "class Calculator { public: int add(int a, int b) { return a + b; } };\n"
        result = parse_file_with_tree_sitter("repo1", "calc.cpp", code, ".cpp")
        assert result.parse_succeeded is True
        assert result.language == "C++"
        names = [s.symbol_name for s in result.symbols]
        assert "Calculator" in names

    def test_parse_malformed_source(self):
        code = "function broken( { return true; }\nclass { }"
        result = parse_file_with_tree_sitter("repo1", "broken.js", code, ".js")
        assert result.parse_succeeded is True
        assert result.language == "JavaScript"

    def test_parse_empty_source(self):
        result = parse_file_with_tree_sitter("repo1", "empty.js", "", ".js")
        assert result.parse_succeeded is True
        assert len(result.symbols) == 0

    def test_parse_unsupported_extension(self):
        result = parse_file_with_tree_sitter("repo1", "file.xyz", "content", ".xyz")
        assert result.parse_succeeded is False
        assert "not available" in result.parse_error

    def test_symbol_has_source_location(self):
        code = "class Calculator {\n  add(a, b) { return a + b; }\n}\n"
        result = parse_file_with_tree_sitter("repo1", "calc.js", code, ".js")
        assert result.parse_succeeded is True
        for sym in result.symbols:
            assert sym.line_start >= 1
            assert sym.line_end >= sym.line_start
            assert sym.file_path == "calc.js"
            assert sym.repository_id == "repo1"

    def test_edge_provenance(self):
        code = "class Animal {}\nclass Dog extends Animal {}\n"
        result = parse_file_with_tree_sitter("repo1", "models.js", code, ".js")
        assert result.parse_succeeded is True
        for edge in result.edges:
            assert edge.extraction_method == FROM_TREE_SITTER
            assert edge.confidence > 0
            assert edge.repository_id == "repo1"


class TestRepositoryParser:
    def test_python_routes_to_ast(self):
        code = "class Calculator:\n    def add(self, a, b):\n        return a + b\n"
        result = parse_file("repo1", "calc.py", code)
        assert result.parse_succeeded is True
        assert result.language == "Python"
        assert result.classes >= 1
        assert any(s.extraction_method == FROM_PYTHON_AST for s in result.symbols)

    def test_javascript_routes_to_tree_sitter(self):
        code = "function greet(name) { return 'Hello ' + name; }\n"
        result = parse_file("repo1", "greet.js", code)
        assert result.parse_succeeded is True
        assert result.language == "JavaScript"
        assert any(s.extraction_method == FROM_TREE_SITTER for s in result.symbols)

    def test_typescript_routes_to_tree_sitter(self):
        code = "function greet(name: string): string { return 'Hello ' + name; }\n"
        result = parse_file("repo1", "greet.ts", code)
        assert result.parse_succeeded is True
        assert result.language == "TypeScript"

    def test_unsupported_extension_returns_failed(self):
        result = parse_file("repo1", "readme.xyz", "some content")
        assert result.parse_succeeded is False
        assert "unsupported" in result.parse_error

    def test_python_fallback_to_tree_sitter_on_error(self):
        code = "class Calculator\n  def add(a, b)\n    return a + b\n"
        result = parse_file("repo1", "broken.py", code)
        assert isinstance(result, NormalizedFile)

    def test_parse_file_never_raises(self):
        try:
            result = parse_file("repo1", "test.py", "")
            assert isinstance(result, NormalizedFile)
        except Exception as exc:
            pytest.fail(f"parse_file raised unexpectedly: {exc}")


class TestSCIPAdapter:
    def test_scip_availability_reports_status(self):
        status = get_scip_availability()
        assert isinstance(status, SCIPAvailability)
        assert status in (SCIPAvailability.NOT_INSTALLED, SCIPAvailability.NO_INDEX)

    def test_is_scip_available_returns_bool(self):
        result = is_scip_available()
        assert isinstance(result, bool)

    def test_find_scip_index_nonexistent_path(self):
        result = find_scip_index("/nonexistent/path/that/does/not/exist")
        assert result is None

    def test_find_scip_index_empty_string(self):
        result = find_scip_index("")
        assert result is None


class TestMultiLanguageRepository:
    def test_mixed_language_parsing(self):
        files = {
            "src/auth.py": "class Authenticator:\n    def authenticate(self):\n        pass\n",
            "src/auth.js": "class AuthService { authenticate() { return true; } }\n",
            "src/auth.go": "type Authenticator interface { Authenticate() bool }\n",
            "src/auth.rs": "trait Authenticator { fn authenticate(&self) -> bool; }\n",
            "src/auth.java": "interface Authenticator { boolean authenticate(); }\n",
        }
        results = {}
        for path, content in files.items():
            results[path] = parse_file("repo1", path, content)
        for path, result in results.items():
            assert result.parse_succeeded is True, f"Failed to parse {path}"
            assert result.language != "Unknown"
            assert len(result.symbols) >= 1, f"No symbols from {path}"

    def test_symbol_provenance_preserved(self):
        code = "class Calculator { add(a, b) { return a + b; } }\n"
        result = parse_file("repo1", "calc.js", code)
        for sym in result.symbols:
            assert sym.repository_id == "repo1"
            assert sym.file_path == "calc.js"
            assert sym.language == "JavaScript"
            assert sym.symbol_name
            assert sym.symbol_type
            assert sym.line_start >= 1
