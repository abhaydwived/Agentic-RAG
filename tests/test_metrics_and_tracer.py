"""
tests/test_metrics_and_tracer.py

Unit and integration tests for Phase 6 Observability and Metrics:
1. Recall@K calculation
2. Reciprocal Rank and Mean Reciprocal Rank (MRR)
3. Context Relevance scoring
4. Answer Faithfulness computation
5. Answer Relevance computation
6. LocalTracer step recording and latency measurement
"""

import time
import unittest

from src.metrics import (
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_mean_reciprocal_rank,
    calculate_context_relevance,
    calculate_answer_faithfulness,
    calculate_answer_relevance,
)
from src.tracer import LocalTracer


class TestMetricsAndTracer(unittest.TestCase):

    def test_recall_at_k(self):
        """Verify Recall@K calculation on top-k matches."""
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
        ground_truth = ["doc_2", "doc_4"]

        # K=1 -> doc_1 (0 hits out of 2) -> 0.0
        self.assertEqual(calculate_recall_at_k(retrieved, ground_truth, k=1), 0.0)

        # K=2 -> doc_1, doc_2 (1 hit out of 2) -> 0.5
        self.assertEqual(calculate_recall_at_k(retrieved, ground_truth, k=2), 0.5)

        # K=4 -> doc_1, doc_2, doc_3, doc_4 (2 hits out of 2) -> 1.0
        self.assertEqual(calculate_recall_at_k(retrieved, ground_truth, k=4), 1.0)

    def test_reciprocal_rank_and_mrr(self):
        """Verify Reciprocal Rank (RR) and MRR computation."""
        # Query 1: first relevant doc at rank 2 -> RR = 0.5
        rr1 = calculate_reciprocal_rank(["wrong_doc", "right_doc"], ["right_doc"])
        self.assertEqual(rr1, 0.5)

        # Query 2: first relevant doc at rank 1 -> RR = 1.0
        rr2 = calculate_reciprocal_rank(["right_doc", "wrong_doc"], ["right_doc"])
        self.assertEqual(rr2, 1.0)

        # MRR across both runs -> (0.5 + 1.0) / 2 = 0.75
        runs = [
            {"retrieved": ["wrong_doc", "right_doc"], "ground_truth": ["right_doc"]},
            {"retrieved": ["right_doc", "wrong_doc"], "ground_truth": ["right_doc"]},
        ]
        mrr = calculate_mean_reciprocal_rank(runs)
        self.assertEqual(mrr, 0.75)

    def test_context_relevance(self):
        """Verify context relevance keyword intersection precision."""
        context = "The annual corporate revenue was $15 million in Q3."
        keywords = ["revenue", "Q3", "million", "missing_word"]

        # 3 out of 4 keywords matched -> 0.75
        score = calculate_context_relevance(context, keywords)
        self.assertEqual(score, 0.75)

    def test_answer_faithfulness(self):
        """Verify sentence-level token overlap for faithfulness."""
        context = "Employees may work from home up to two days per week with manager approval."
        faithful_answer = "Employees can work from home two days each week with manager approval."
        unfaithful_answer = "Employees receive unlimited paid vacation and free daily lunches."

        score_faithful = calculate_answer_faithfulness(faithful_answer, context)
        score_unfaithful = calculate_answer_faithfulness(unfaithful_answer, context)

        self.assertGreater(score_faithful, 0.8)
        self.assertLess(score_unfaithful, 0.5)

    def test_answer_relevance(self):
        """Verify question term alignment in generated answer."""
        query = "What is AAPL stock price?"
        relevant_answer = "The stock price of AAPL is $180."
        irrelevant_answer = "The weather in Paris is sunny."

        rel_score = calculate_answer_relevance(relevant_answer, query)
        irrel_score = calculate_answer_relevance(irrelevant_answer, query)

        self.assertGreater(rel_score, 0.6)
        self.assertLess(irrel_score, 0.3)

    def test_local_tracer(self):
        """Verify tracing execution steps, tool records, and latencies."""
        tracer = LocalTracer()
        trace = tracer.start_trace(query="Test query", session_id="test_sess")

        # Step 1
        s1 = tracer.record_step_start("document_retrieval", inputs={"query": "test"})
        time.sleep(0.01)
        tracer.record_step_end(s1, outputs={"chunks": 3}, tokens=50)

        # Step 2
        s2 = tracer.record_step_start("answer_generation", inputs={})
        time.sleep(0.01)
        tracer.record_step_end(s2, outputs={"ans": "test"}, tokens=120)

        finished = tracer.finish_trace(
            llm_calls=2,
            tool_calls=["search_knowledge_base"],
            retrieval_chunks=3,
            iterations=1,
        )

        self.assertEqual(finished.query, "Test query")
        self.assertEqual(finished.llm_call_count, 2)
        self.assertEqual(finished.total_tokens, 170)
        self.assertGreater(finished.total_latency_seconds, 0.01)
        self.assertEqual(len(finished.steps), 2)


if __name__ == "__main__":
    unittest.main()

