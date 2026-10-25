"""
src/metrics.py

Evaluation Metrics Engine:
Calculates empirical retrieval and generation quality scores:
- Recall@K: Proportion of relevant chunks retrieved in the top K.
- Mean Reciprocal Rank (MRR): 1 / rank of the first relevant chunk.
- Context Relevance: Precision of retrieved chunks against ground truth concepts.
- Answer Faithfulness: Claim-level precision against provided context.
- Answer Relevance: Semantic alignment between question and generated response.
- End-to-End Latency.
"""

import re
from typing import Any, Dict, List, Set


def calculate_recall_at_k(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: List[str],
    k: int = 3,
) -> float:
    """
    Computes Recall@K:
    Recall@K = |Retrieved[:k] ∩ GroundTruth| / |GroundTruth|
    """
    if not ground_truth_chunk_ids:
        return 1.0

    top_k_retrieved = set(retrieved_chunk_ids[:k])
    relevant_set = set(ground_truth_chunk_ids)

    hits = len(top_k_retrieved.intersection(relevant_set))
    return float(hits / len(relevant_set))


def calculate_reciprocal_rank(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: List[str],
) -> float:
    """
    Computes Reciprocal Rank (RR):
    RR = 1 / rank (where rank is 1-indexed position of the FIRST relevant item).
    """
    if not ground_truth_chunk_ids:
        return 1.0

    relevant_set = set(ground_truth_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_set:
            return float(1.0 / rank)
    return 0.0


def calculate_mean_reciprocal_rank(
    retrieval_runs: List[Dict[str, List[str]]],
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) across multiple queries.
    Each item in retrieval_runs:
      {"retrieved": [...], "ground_truth": [...]}
    """
    if not retrieval_runs:
        return 0.0

    rrs = [
        calculate_reciprocal_rank(run["retrieved"], run["ground_truth"])
        for run in retrieval_runs
    ]
    return float(sum(rrs) / len(rrs))


def calculate_context_relevance(
    context_text: str,
    target_keywords: List[str],
) -> float:
    """
    Computes keyword/concept overlap precision:
    Proportion of target gold concepts present in the retrieved context.
    """
    if not target_keywords:
        return 1.0
    if not context_text:
        return 0.0

    context_lower = context_text.lower()
    matches = sum(1 for kw in target_keywords if kw.lower() in context_lower)
    return float(matches / len(target_keywords))


def calculate_answer_faithfulness(
    answer_text: str,
    context_text: str,
) -> float:
    """
    Computes the percentage of factual sentences in the answer that are
    supported by the context (unsupported sentences indicate potential hallucination).
    """
    if not answer_text:
        return 0.0
    if not context_text:
        return 0.0

    # Split answer into declarative sentences
    sentences = [s.strip() for s in re.split(r"[.!?]", answer_text) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    context_words = set(re.findall(r"\w+", context_text.lower()))
    supported_count = 0

    for sent in sentences:
        sent_words = [w for w in re.findall(r"\w+", sent.lower()) if len(w) > 3]
        if not sent_words:
            supported_count += 1
            continue
        # Check token containment proportion
        overlap = sum(1 for w in sent_words if w in context_words)
        ratio = overlap / len(sent_words)
        if ratio >= 0.6:  # 60% substantive token overlap threshold
            supported_count += 1

    return float(supported_count / len(sentences))


def calculate_answer_relevance(
    answer_text: str,
    query_text: str,
) -> float:
    """
    Computes query-answer semantic alignment based on core entity and question term containment.
    """
    if not query_text or not answer_text:
        return 0.0

    query_tokens = [w for w in re.findall(r"\w+", query_text.lower()) if len(w) > 2]
    if not query_tokens:
        return 1.0

    answer_lower = answer_text.lower()
    matches = sum(1 for tok in query_tokens if tok in answer_lower)
    return float(matches / len(query_tokens))

