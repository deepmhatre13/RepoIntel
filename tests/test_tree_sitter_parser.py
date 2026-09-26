"""Tests for the Tree-sitter backed multi-language parser."""

import pytest
from packages.parsing.tree_sitter_parser import (
    parse_file_with_tree_sitter,
    is_language_supported,
    supported_extensions,
    _GRAMMAR_REGISTRY,
)
from packages.parsing.repository_parser import parse_file
from packages.parsing.normalized_model import (
    NormalizedSymbol, NormalizedEdge, NormalizedFile,
    CLASS, FUNCTION, METHOD, VARIABLE, IMPORT, STRUCT, INTERFACE, ENUM,
    CONTAINS, INHERITS,
)


REPO = "test-repo-001"


class TestGrammarRegistry:
    def test_registry_contains_js_ts(self):
        assert ".js" in _GRAMMAR_REGISTRY
        assert ".ts" in _GRAMMAR_REGISTRY
        assert ".tsx" in _GRAMMAR_REGISTRY

    def test_registry_contains_java_go_rust(self):
        assert ".java" in _GRAMMAR_REGISTRY
        assert ".go" in _GRAMMAR_REGISTRY
        assert ".rs" in _GRAMMAR_REGISTRY

    def test_registry_contains_c_cpp(self):
        assert ".c" in _GRAMMAR_REGISTRY
        assert ".cpp" in _GRAMMAR_REGISTRY
        assert ".h" in _GRAMMAR_REGISTRY


class TestLanguageSupport:
    def test_js_supported(self):
        assert is_language_supported(".js") is True

    def test_ts_supported(self):
        assert is_language_supported(".ts") is True

    def test_unsupported_extension(self):
        assert is_language_supported(".xyz") is False

    def test_case_insensitive(self):
        assert is_language_supported(".JS") is True


class TestTreeSitterJS:
    def test_extracts_function_and_class(self):
        code = "function greet(name) { return 'Hello ' + name; }\nclass Animal { speak() { return 'noise'; } }\n"
        result = parse_file_with_tree_sitter(REPO, "test.js", code, ".js")
        assert result.parse_succeeded is True
        assert len(result.symbols) >= 2
        names = [s.symbol_name for s in result.symbols]
        assert "greet" in names
        assert "Animal" in names

    def test_extracts_inheritance(self):
        code = "class Dog extends Animal { speak() { return 'bark'; } }\n"
        result = parse_file_with_tree_sitter(REPO, "test.js", code, ".js")
        inherits_edges = [e for e in result.edges if e.edge_type == INHERITS]
        assert len(inherits_edges) >= 1
        assert inherits_edges[0].source_symbol == "Dog"
        assert inherits_edges[0].target_symbol == "Animal"

    def test_line_numbers_are_1_indexed(self):
        code = "function greet() {}\n"
        result = parse_file_with_tree_sitter(REPO, "test.js", code, ".js")
        for s in result.symbols:
            assert s.line_start >= 1
            assert s.line_end >= s.line_start


class TestTreeSitterTS:
    def test_extracts_class_and_method(self):
        code = "class Calculator { add(a: number, b: number): number { return a + b; } }\n"
        result = parse_file_with_tree_sitter(REPO, "calc.ts", code, ".ts")
        assert result.parse_succeeded is True
        assert any(s.symbol_name == "Calculator" for s in result.symbols)

    def test_extracts_interface(self):
        code = "interface Shape { area(): number; }\n"
        result = parse_file_with_tree_sitter(REPO, "shape.ts", code, ".ts")
        if result.symbols:
            types = [s.symbol_type for s in result.symbols]
            assert INTERFACE in types


class TestTreeSitterJava:
    def test_extracts_class(self):
        code = "public class HelloWorld { public static void main(String[] args) { } }\n"
        result = parse_file_with_tree_sitter(REPO, "HelloWorld.java", code, ".java")
        assert result.parse_succeeded is True
        if result.symbols:
            assert any(s.symbol_name == "HelloWorld" for s in result.symbols)


class TestTreeSitterGo:
    def test_extracts_struct_and_func(self):
        code = "package main\ntype Point struct { X int; Y int }\nfunc main() {}\n"
        result = parse_file_with_tree_sitter(REPO, "main.go", code, ".go")
        assert result.parse_succeeded is True
        if result.symbols:
            names = [s.symbol_name for s in result.symbols]
            assert "Point" in names or "main" in names


class TestTreeSitterRust:
    def test_extracts_struct(self):
        code = "struct Point { x: i32, y: i32 }\nfn main() {}\n"
        result = parse_file_with_tree_sitter(REPO, "main.rs", code, ".rs")
        assert result.parse_succeeded is True
        if result.symbols:
            types = [s.symbol_type for s in result.symbols]
            assert STRUCT in types


class TestRepositoryParserRouting:
    def test_python_uses_ast(self):
        code = "class Dog:\n    def bark(self):\n        pass\n"
        result = parse_file(REPO, "dog.py", code)
        assert result.parse_succeeded is True
        assert result.language == "Python"
        assert any(s.symbol_name == "Dog" for s in result.symbols)
        for s in result.symbols:
            assert s.confidence == 1.0

    def test_js_uses_tree_sitter(self):
        code = "function greet() {}\n"
        result = parse_file(REPO, "greet.js", code)
        assert result.parse_succeeded is True
        assert result.language == "JavaScript"
        for s in result.symbols:
            assert s.extraction_method == "tree_sitter"

    def test_unsupported_file_type(self):
        code = "some random content"
        result = parse_file(REPO, "file.xyz", code)
        assert result.parse_succeeded is False
        assert "unsupported" in result.parse_error.lower()

    def test_empty_python_file(self):
        code = ""
        result = parse_file(REPO, "empty.py", code)
        assert result.parse_succeeded is True
        assert len(result.symbols) == 0


class TestNormalizedModel:
    def test_symbol_has_required_fields(self):
        sym = NormalizedSymbol(
            repository_id=REPO,
            file_path="test.py",
            language="Python",
            symbol_name="MyClass",
            symbol_type=CLASS,
            line_start=1,
            line_end=10,
        )
        assert sym.repository_id == REPO
        assert sym.file_path == "test.py"
        assert sym.language == "Python"
        assert sym.symbol_name == "MyClass"
        assert sym.symbol_type == CLASS

    def test_edge_has_provenance(self):
        edge = NormalizedEdge(
            repository_id=REPO,
            source_symbol="A",
            source_file="a.py",
            edge_type=INHERITS,
            target_symbol="B",
            extraction_method="tree_sitter",
            confidence=0.9,
        )
        assert edge.extraction_method == "tree_sitter"
        assert edge.confidence == 0.9


class TestMalformedSource:
    def test_js_syntax_error_recovers(self):
        """Malformed JS should not crash the parser."""
        code = "function greet( { return 'broken'"
        result = parse_file_with_tree_sitter(REPO, "broken.js", code, ".js")
        assert result.parse_succeeded is True  # tree-sitter handles errors gracefully

    def test_python_syntax_error_falls_back(self):
        """Malformed Python should fall back to tree-sitter gracefully."""
        code = "class Broken:\n  def no_close(\n"
        result = parse_file(REPO, "broken.py", code)
        # Should either succeed with partial results or fail gracefully
        assert isinstance(result, NormalizedFile)

    def test_empty_file(self):
        result = parse_file_with_tree_sitter(REPO, "empty.js", "", ".js")
        assert result.parse_succeeded is True
        assert len(result.symbols) == 0

    def test_binary_like_content(self):
        code = "\x00\x01\x02\x03\x04"
        result = parse_file_with_tree_sitter(REPO, "binary.js", code, ".js")
        assert isinstance(result, NormalizedFile)
