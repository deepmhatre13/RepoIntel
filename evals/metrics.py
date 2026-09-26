from typing import List
import math

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


def calculate_ndcg_at_k(retrieved_items: List[str], expected_items: List[str], k: int = 10) -> float:
    """Normalized discounted cumulative gain (standard binary-relevance form).

    DCG  = sum(rel_i / log2(i + 1)) for i = 1..k (i is the 1-based rank)
    IDCG = DCG of the ideal ranking (all relevant items at the top)

    Binary relevance: rel_i = 1 if item i is relevant else 0. When there are
    zero expected items the metric is undefined; returns 0.0 (callers that need
    a distinct state should filter unanswerable queries before calling).
    """
    if not expected_items:
        return 0.0
    expected_set = set(expected_items)
    dcg = 0.0
    for idx, item in enumerate(retrieved_items[:k], start=1):
        if item in expected_set:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
