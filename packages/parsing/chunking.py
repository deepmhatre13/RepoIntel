import hashlib
from dataclasses import dataclass
from typing import List, Optional
from packages.parsing.python_ast import SymbolNode

@dataclass
class CodeChunk:
    repo_id: str
    file_path: str
    symbol_name: Optional[str]
    line_start: int
    line_end: int
    content: str
    content_hash: str
    language: str

MAX_CHUNK_LINES = 200
CHUNK_OVERLAP_LINES = 10
FALLBACK_CHUNK_LINES = 60

def create_ast_aware_chunks(repo_id: str, file_path: str, content: str, language: str, symbols: List[SymbolNode]) -> List[CodeChunk]:
    lines = content.splitlines()
    if not lines:
        return []

    chunks = []
    covered_lines = set()

    def append_span(start: int, end: int, symbol_name: Optional[str], chunk_size: int):
        curr = start
        while curr <= end:
            sub_end = min(curr + chunk_size - 1, end)
            sub_content = "\n".join(lines[curr - 1:sub_end])
            if sub_content.strip():
                chash = hashlib.sha256(sub_content.encode("utf-8")).hexdigest()
                chunks.append(CodeChunk(
                    repo_id=repo_id,
                    file_path=file_path,
                    symbol_name=symbol_name,
                    line_start=curr,
                    line_end=sub_end,
                    content=sub_content,
                    content_hash=chash,
                    language=language,
                ))
            if sub_end == end:
                break
            curr = sub_end - CHUNK_OVERLAP_LINES + 1

    # 1. Keep AST-derived functions/classes as structural chunks.
    for sym in symbols:
        start = sym.line_start
        end = sym.line_end
        if start < 1 or end < start or start > len(lines):
            continue
        end = min(end, len(lines))
        append_span(start, end, sym.symbol_name, MAX_CHUNK_LINES)
        covered_lines.update(range(start, end + 1))

    # 2. Chunk only contiguous non-symbol lines; do not let fallback windows
    # swallow already-complete function/class chunks.
    total_lines = len(lines)
    curr = 1
    while curr <= total_lines:
        if curr in covered_lines:
            curr += 1
            continue
        span_start = curr
        while curr <= total_lines and curr not in covered_lines:
            curr += 1
        append_span(span_start, curr - 1, None, FALLBACK_CHUNK_LINES)

    return chunks
