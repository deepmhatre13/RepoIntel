from typing import List, Dict, Set
from packages.knowledge.vector_store import RetrievedEvidenceItem

class ContextAssembler:
    def __init__(self, max_context_chars: int = 12000):
        self.max_context_chars = max_context_chars

    def assemble(self, items: List[RetrievedEvidenceItem]) -> List[RetrievedEvidenceItem]:
        seen_keys: Set[str] = set()
        unique_items = []
        total_chars = 0

        # Sort items by relevance score descending
        sorted_items = sorted(items, key=lambda x: x.relevance_score, reverse=True)

        for item in sorted_items:
            key = f"{item.file_path}:{item.line_start}:{item.line_end}"
            if key in seen_keys:
                continue

            item_len = len(item.content)
            if total_chars + item_len > self.max_context_chars and unique_items:
                break

            seen_keys.add(key)
            unique_items.append(item)
            total_chars += item_len

        return unique_items
