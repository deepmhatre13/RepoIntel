from packages.parsing.chunking import create_ast_aware_chunks
from packages.parsing.python_ast import SymbolNode, parse_python_file


def test_fallback_chunks_only_uncovered_lines_not_function_body():
    content = "\n".join([
        "def start():",
        "    configure()",
        "    return True",
        "",
        "# module-level configuration",
        "DEFAULT_TIMEOUT = 30",
        "",
    ])
    symbol = SymbolNode(
        file_path="app.py",
        symbol_name="start",
        symbol_type="function",
        line_start=1,
        line_end=3,
        signature="def start():",
    )

    chunks = create_ast_aware_chunks("repo-1", "app.py", content, "Python", [symbol])

    function_chunks = [chunk for chunk in chunks if chunk.symbol_name == "start"]
    fallback_chunks = [chunk for chunk in chunks if chunk.symbol_name is None]
    assert len(function_chunks) == 1
    assert (function_chunks[0].line_start, function_chunks[0].line_end) == (1, 3)
    assert fallback_chunks
    assert all(chunk.line_start >= 4 for chunk in fallback_chunks)
    assert all("def start" not in chunk.content for chunk in fallback_chunks)


def test_oversized_symbol_chunks_overlap_and_stay_bounded():
    content = "\n".join(f"    statement_{line}()" for line in range(1, 206))
    symbol = SymbolNode(
        file_path="large.py",
        symbol_name="large_function",
        symbol_type="function",
        line_start=1,
        line_end=205,
        signature="def large_function():",
    )

    chunks = create_ast_aware_chunks("repo-1", "large.py", content, "Python", [symbol])

    assert len(chunks) == 2
    assert all(chunk.symbol_name == "large_function" for chunk in chunks)
    assert all(chunk.line_end - chunk.line_start + 1 <= 200 for chunk in chunks)
    assert chunks[0].line_end - chunks[1].line_start + 1 == 10


def test_decorators_are_included_in_python_ast_chunk_boundaries():
    content = "@register_service\ndef create_service():\n    return Service()\n"
    parse_result = parse_python_file("service.py", content)

    chunks = create_ast_aware_chunks(
        "repo-1", "service.py", content, "Python", parse_result.symbols
    )

    function_chunk = next(chunk for chunk in chunks if chunk.symbol_name == "create_service")
    assert function_chunk.line_start == 1
    assert function_chunk.content.startswith("@register_service")
    assert "def create_service" in function_chunk.content