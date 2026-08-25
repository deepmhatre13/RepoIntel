import hashlib
from dataclasses import dataclass
from typing import Optional

@dataclass
class GenericFileMetadata:
    file_path: str
    content_hash: str
    size_bytes: int
    language: str

def parse_generic_fallback(file_path: str, content: str, language: str = "Unknown") -> GenericFileMetadata:
    content_bytes = content.encode("utf-8", errors="ignore")
    content_hash = hashlib.sha256(content_bytes).hexdigest()
    return GenericFileMetadata(
        file_path=file_path,
        content_hash=content_hash,
        size_bytes=len(content_bytes),
        language=language
    )
