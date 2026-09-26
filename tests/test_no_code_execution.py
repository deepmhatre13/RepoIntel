"""Targeted test verifying that repository ingestion, parsing, and indexing NEVER execute repository code.
"""
import os
import tempfile
import pytest
from packages.parsing.python_ast import parse_python_file


def test_parsing_never_executes_repository_code():
    """Create a repository file containing executable code with a side effect (creating a marker file).
    Verify that parsing the file using AST parser extracts symbols without creating the marker file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        marker_path = os.path.join(tmpdir, "MALICIOUS_EXECUTION_MARKER.txt")
        malicious_file = os.path.join(tmpdir, "malicious.py")
        
        # Code that creates a marker file if executed or imported
        malicious_code = f"""import os

# Side-effect on import/execution
with open(r"{marker_path}", "w") as f:
    f.write("EXECUTED")

def dangerous_function():
    return "malicious"
"""
        with open(malicious_file, "w", encoding="utf-8") as f:
            f.write(malicious_code)
            
        result = parse_python_file(malicious_file, malicious_code)
        
        # 1. Parsing succeeded and extracted symbols
        assert result.parse_succeeded is True
        assert len(result.symbols) >= 1
        
        # 2. Side-effect marker was NEVER created (code was parsed, not executed)
        assert not os.path.exists(marker_path), "CRITICAL SECURITY VIOLATION: Repository code was executed during parsing!"
