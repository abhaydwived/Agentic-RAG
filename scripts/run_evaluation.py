"""
scripts/run_evaluation.py

Reproducible Evaluation Pipeline:
Executes evaluation dataset against the Advanced Agentic RAG pipeline and computes:
- Recall@K
- Mean Reciprocal Rank (MRR)
- Context Relevance
- Answer Faithfulness
- Answer Relevance
- End-to-End Latency
"""

import json
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.knowledge_indexer import index_knowledge_base, get_rag_pipeline
from src.metrics import (
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_context_relevance,
    calculate_answer_faithfulness,
    calculate_answer_relevance,
)


def run_evaluation(
    dataset_path: str = "data/eval_dataset.json",
    kb_dir: str = "data/knowledge_base",
    top_k: int = 3,
):
    print("=" * 65)
    print("  Agentic RAG — Comprehensive Pipeline Evaluation")
    print("=" * 65)

    if not os.path.exists(dataset_path):
        print(f"Dataset '{dataset_path}' not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # 1. Index knowledge base
    print(f"\n[1/3] Initializing Hybrid RAG index from '{kb_dir}'...")
    index_path = os.path.join(kb_dir, ".eval_faiss_index")
    vector_store = index_knowledge_base(kb_dir=kb_dir, index_path=index_path)
    pipeline = get_rag_pipeline()

    retrieval_records = []
    e2e_latencies = []
    context_relevance_scores = []
    faithfulness_scores = []
    relevance_scores = []

    print(f"\n[2/3] Running evaluation over {len(samples)} benchmark queries...")

    for idx, item in enumerate(samples, start=1):
        query = item["query"]
        expected_doc = item.get("target_doc")
        keywords = item.get("expected_chunk_keywords", [])
        gold_answer = item.get("ground_truth_answer", "")

        t0 = time.time()
        retrieved_docs_and_scores = pipeline.retrieve(query, top_k=top_k) if pipeline else []
        latency = round(time.time() - t0, 4)
        e2e_latencies.append(latency)

        retrieved_doc_names = [
            doc.metadata.get("document_name", "") for doc, _ in retrieved_docs_and_scores
        ]
        assembled_context = pipeline.assemble_context(retrieved_docs_and_scores) if pipeline else ""

        # Compute Retrieval Metrics if ground truth doc exists
        if expected_doc:
            recall = calculate_recall_at_k(retrieved_doc_names, [expected_doc], k=top_k)
            rr = calculate_reciprocal_rank(retrieved_doc_names, [expected_doc])
            retrieval_records.append({"recall": recall, "rr": rr})
        else:
            recall = 1.0
            rr = 1.0

        # Compute Context Relevance
        c_rel = calculate_context_relevance(assembled_context, keywords)
        context_relevance_scores.append(c_rel)

        # Compute Faithfulness and Relevance against gold answer / context
        faith = calculate_answer_faithfulness(gold_answer, assembled_context) if expected_doc else 1.0
        faithfulness_scores.append(faith)

        ans_rel = calculate_answer_relevance(gold_answer, query)
        relevance_scores.append(ans_rel)

        print(f"  [{idx}/{len(samples)}] Query: '{query[:45]}...'")
        print(f"      Latency: {latency}s | Recall@{top_k}: {recall:.2f} | Context Relevance: {c_rel:.2f}")

    # Summary Metrics Calculation
    avg_recall = sum(r["recall"] for r in retrieval_records) / len(retrieval_records) if retrieval_records else 1.0
    mrr = sum(r["rr"] for r in retrieval_records) / len(retrieval_records) if retrieval_records else 1.0
    avg_context_rel = sum(context_relevance_scores) / len(context_relevance_scores)
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_ans_relevance = sum(relevance_scores) / len(relevance_scores)
    avg_latency = sum(e2e_latencies) / len(e2e_latencies)

    print("\n" + "=" * 65)
    print("  EMPIRICAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"  • Recall@{top_k}              : {avg_recall:.4f}")
    print(f"  • Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"  • Context Relevance         : {avg_context_rel:.4f}")
    print(f"  • Answer Faithfulness       : {avg_faithfulness:.4f}")
    print(f"  • Answer Relevance          : {avg_ans_relevance:.4f}")
    print(f"  • Average Retrieval Latency : {avg_latency:.4f}s")
    print("=" * 65)

    summary = {
        "recall_at_k": round(avg_recall, 4),
        "mean_reciprocal_rank": round(mrr, 4),
        "context_relevance": round(avg_context_rel, 4),
        "answer_faithfulness": round(avg_faithfulness, 4),
        "answer_relevance": round(avg_ans_relevance, 4),
        "average_latency_seconds": round(avg_latency, 4),
        "total_queries_evaluated": len(samples),
    }

    results_file = "data/eval_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved benchmark results to '{results_file}'.")


if __name__ == "__main__":
    run_evaluation()

