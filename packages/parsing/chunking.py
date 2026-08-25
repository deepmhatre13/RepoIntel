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

def create_ast_aware_chunks(repo_id: str, file_path: str, content: str, language: str, symbols: List[SymbolNode]) -> List[CodeChunk]:
    lines = content.splitlines()
    if not lines:
        return []

    chunks = []
    covered_lines = set()

    # 1. Symbol-based chunking
    for sym in symbols:
        start = sym.line_start
        end = sym.line_end
        
        # Split large symbols if line count exceeds MAX_CHUNK_LINES
        if end - start + 1 > MAX_CHUNK_LINES:
            curr = start
            while curr <= end:
                sub_end = min(curr + MAX_CHUNK_LINES - 1, end)
                sub_content = "\n".join(lines[curr-1:sub_end])
                chash = hashlib.sha256(sub_content.encode("utf-8")).hexdigest()
                chunks.append(CodeChunk(
                    repo_id=repo_id,
                    file_path=file_path,
                    symbol_name=sym.symbol_name,
                    line_start=curr,
                    line_end=sub_end,
                    content=sub_content,
                    content_hash=chash,
                    language=language
                ))
                for idx in range(curr, sub_end + 1):
                    covered_lines.add(idx)
                curr = sub_end + 1
        else:
            sub_content = "\n".join(lines[start-1:end])
            chash = hashlib.sha256(sub_content.encode("utf-8")).hexdigest()
            chunks.append(CodeChunk(
                repo_id=repo_id,
                file_path=file_path,
                symbol_name=sym.symbol_name,
                line_start=start,
                line_end=end,
                content=sub_content,
                content_hash=chash,
                language=language
            ))
            for idx in range(start, end + 1):
                covered_lines.add(idx)

    # 2. Sliding window fallback for uncovered lines
    window = 60
    overlap = 10
    total_lines = len(lines)
    
    curr = 1
    while curr <= total_lines:
        end = min(curr + window - 1, total_lines)
        # Check if window covers non-symbol code
        if any(idx not in covered_lines for idx in range(curr, end + 1)):
            sub_content = "\n".join(lines[curr-1:end])
            chash = hashlib.sha256(sub_content.encode("utf-8")).hexdigest()
            chunks.append(CodeChunk(
                repo_id=repo_id,
                file_path=file_path,
                symbol_name=None,
                line_start=curr,
                line_end=end,
                content=sub_content,
                content_hash=chash,
                language=language
            ))
        curr += window - overlap

    return chunks
