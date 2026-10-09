import ast
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SymbolNode:
    file_path: str
    symbol_name: str
    symbol_type: str  # "function", "class", "method"
    line_start: int
    line_end: int
    signature: str
    docstring: Optional[str] = None
    parent_class: Optional[str] = None

@dataclass
class CallEdge:
    source_symbol: str
    source_file: str
    edge_type: str  # "calls", "inherits", "imports"
    target_symbol: str
    target_file: Optional[str] = None

@dataclass
class PythonParseResult:
    file_path: str
    symbols: List[SymbolNode] = field(default_factory=list)
    edges: List[CallEdge] = field(default_factory=list)
    parse_succeeded: bool = True
    parse_error: Optional[str] = None

class PythonASTVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, source_code: str):
        self.file_path = file_path
        self.source_lines = source_code.splitlines()
        self.symbols: List[SymbolNode] = []
        self.edges: List[CallEdge] = []
        self.current_class: Optional[str] = None
        self.current_symbol: Optional[str] = None

    @staticmethod
    def _declaration_start(node) -> int:
        decorator_lines = [
            decorator.lineno
            for decorator in getattr(node, "decorator_list", [])
            if hasattr(decorator, "lineno")
        ]
        return min([node.lineno, *decorator_lines])

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.edges.append(CallEdge(
                source_symbol=self.current_symbol or "<file>",
                source_file=self.file_path,
                edge_type="imports",
                target_symbol=alias.name
            ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            full_target = f"{module}.{alias.name}" if module else alias.name
            self.edges.append(CallEdge(
                source_symbol=self.current_symbol or "<file>",
                source_file=self.file_path,
                edge_type="imports",
                target_symbol=full_target
            ))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        line_start = self._declaration_start(node)
        line_end = getattr(node, "end_lineno", line_start)
        docstring = ast.get_docstring(node)
        
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
                self.edges.append(CallEdge(
                    source_symbol=node.name,
                    source_file=self.file_path,
                    edge_type="inherits",
                    target_symbol=b.id
                ))
        
        sig = f"class {node.name}({', '.join(bases)})"
        sym = SymbolNode(
            file_path=self.file_path,
            symbol_name=node.name,
            symbol_type="class",
            line_start=line_start,
            line_end=line_end,
            signature=sig,
            docstring=docstring,
            parent_class=None
        )
        self.symbols.append(sym)

        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_function(node, is_async=True)

    def _handle_function(self, node, is_async: bool = False):
        line_start = self._declaration_start(node)
        line_end = getattr(node, "end_lineno", line_start)
        docstring = ast.get_docstring(node)
        
        args = [arg.arg for arg in node.args.args]
        prefix = "async def " if is_async else "def "
        sig = f"{prefix}{node.name}({', '.join(args)})"

        sym_type = "method" if self.current_class else "function"
        full_name = f"{self.current_class}.{node.name}" if self.current_class else node.name

        sym = SymbolNode(
            file_path=self.file_path,
            symbol_name=full_name,
            symbol_type=sym_type,
            line_start=line_start,
            line_end=line_end,
            signature=sig,
            docstring=docstring,
            parent_class=self.current_class
        )
        self.symbols.append(sym)

        old_symbol = self.current_symbol
        self.current_symbol = full_name
        self.generic_visit(node)
        self.current_symbol = old_symbol

    def visit_Call(self, node: ast.Call):
        target_name = None
        if isinstance(node.func, ast.Name):
            target_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            target_name = node.func.attr

        if target_name and self.current_symbol:
            self.edges.append(CallEdge(
                source_symbol=self.current_symbol,
                source_file=self.file_path,
                edge_type="calls",
                target_symbol=target_name
            ))
        self.generic_visit(node)

def parse_python_file(file_path: str, content: str) -> PythonParseResult:
    try:
        tree = ast.parse(content, filename=file_path)
        visitor = PythonASTVisitor(file_path, content)
        visitor.visit(tree)
        return PythonParseResult(
            file_path=file_path,
            symbols=visitor.symbols,
            edges=visitor.edges,
            parse_succeeded=True
        )
    except Exception as e:
        return PythonParseResult(
            file_path=file_path,
            symbols=[],
            edges=[],
            parse_succeeded=False,
            parse_error=str(e)
        )
