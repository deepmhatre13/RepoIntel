from typing import List, Dict, Any

def calculate_recall_at_k(retrieved_items: List[str], expected_items: List[str], k: int = 5) -> float:
    if not expected_items:
        return 1.0
    retrieved_k = set(retrieved_items[:k])
    expected_set = set(expected_items)
    intersection = retrieved_k.intersection(expected_set)
    return len(intersection) / len(expected_set)

def calculate_precision_at_k(retrieved_items: List[str], expected_items: List[str], k: int = 5) -> float:
    if not retrieved_items or k == 0:
        return 0.0
    retrieved_k = set(retrieved_items[:k])
    expected_set = set(expected_items)
    intersection = retrieved_k.intersection(expected_set)
    return len(intersection) / min(k, len(retrieved_items))

def calculate_mrr(retrieved_items: List[str], expected_items: List[str]) -> float:
    expected_set = set(expected_items)
    for idx, item in enumerate(retrieved_items, start=1):
        if item in expected_set:
            return 1.0 / idx
    return 0.0

def calculate_supported_claim_ratio(total_claims: int, supported_claims: int) -> float:
    if total_claims == 0:
        return 1.0
    return supported_claims / total_claims
