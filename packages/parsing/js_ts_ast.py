import re
from dataclasses import dataclass, field
from typing import List, Optional
from packages.parsing.python_ast import SymbolNode, CallEdge

@dataclass
class JSParseResult:
    file_path: str
    symbols: List[SymbolNode] = field(default_factory=list)
    edges: List[CallEdge] = field(default_factory=list)
    parse_succeeded: bool = True
    parse_error: Optional[str] = None

# Regex patterns for basic TS/JS structural extraction
FUNC_PATTERN = re.compile(r'(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)\s*\(([^)]*)\)')
ARROW_PATTERN = re.compile(r'(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>')
CLASS_PATTERN = re.compile(r'(?:export\s+)?class\s+([A-Za-z0-9_$]+)(?:\s+extends\s+([A-Za-z0-9_$]+))?')
IMPORT_PATTERN = re.compile(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]')
REQUIRE_PATTERN = re.compile(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)')

def parse_js_ts_file(file_path: str, content: str) -> JSParseResult:
    symbols = []
    edges = []
    lines = content.splitlines()

    for idx, line in enumerate(lines, start=1):
        # 1. Imports
        for imp in IMPORT_PATTERN.findall(line):
            edges.append(CallEdge(source_symbol="<file>", source_file=file_path, edge_type="imports", target_symbol=imp))
        for req in REQUIRE_PATTERN.findall(line):
            edges.append(CallEdge(source_symbol="<file>", source_file=file_path, edge_type="imports", target_symbol=req))

        # 2. Classes
        class_match = CLASS_PATTERN.search(line)
        if class_match:
            cls_name = class_match.group(1)
            extends_cls = class_match.group(2)
            if extends_cls:
                edges.append(CallEdge(source_symbol=cls_name, source_file=file_path, edge_type="inherits", target_symbol=extends_cls))
            symbols.append(SymbolNode(
                file_path=file_path,
                symbol_name=cls_name,
                symbol_type="class",
                line_start=idx,
                line_end=min(idx + 50, len(lines)),
                signature=f"class {cls_name}" + (f" extends {extends_cls}" if extends_cls else "")
            ))

        # 3. Functions
        func_match = FUNC_PATTERN.search(line)
        if func_match:
            func_name = func_match.group(1)
            params = func_match.group(2)
            symbols.append(SymbolNode(
                file_path=file_path,
                symbol_name=func_name,
                symbol_type="function",
                line_start=idx,
                line_end=min(idx + 30, len(lines)),
                signature=f"function {func_name}({params})"
            ))

        # 4. Arrow Functions
        arrow_match = ARROW_PATTERN.search(line)
        if arrow_match:
            func_name = arrow_match.group(1)
            params = arrow_match.group(2)
            symbols.append(SymbolNode(
                file_path=file_path,
                symbol_name=func_name,
                symbol_type="function",
                line_start=idx,
                line_end=min(idx + 30, len(lines)),
                signature=f"const {func_name} = ({params}) =>"
            ))

    return JSParseResult(
        file_path=file_path,
        symbols=symbols,
        edges=edges,
        parse_succeeded=True
    )
