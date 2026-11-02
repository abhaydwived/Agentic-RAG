"""
tests/test_evaluation_layer.py

Unit and integration tests for Phase 3 Evaluation Layer:
1. Retrieval relevance assessment
2. Context relevance checking
3. Answer groundedness verification & hallucination detection
4. Source citation validation
5. Insufficient information detection (graceful refusal rather than hallucination)
"""

import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from src.evaluator import QualityEvaluator, EvaluationReport


class TestEvaluationLayer(unittest.TestCase):

    def setUp(self):
        self.mock_llm = MagicMock()
        self.evaluator = QualityEvaluator(self.mock_llm)

    def test_context_relevance_empty_or_missing(self):
        """Verify that empty or default missing context is flagged as insufficient."""
        report = self.evaluator.assess_context_relevance(
            query="What is the internal travel policy?",
            context="No relevant internal documents found.",
        )
        self.assertFalse(report["is_sufficient"])
        self.assertEqual(report["relevance_score"], 0.0)

    def test_grounded_answer_evaluation(self):
        """Verify passing report for a grounded answer with valid citations."""
        # 1. Relevance call
        # 2. Groundedness call
        self.mock_llm.invoke.side_effect = [
            AIMessage(content='{"is_sufficient": true, "relevance_score": 0.95, "missing_information": ""}'),
            AIMessage(content='{"groundedness_score": 0.98, "hallucination_detected": false, "citations_valid": true, "is_valid": true, "feedback": ""}'),
        ]

        query = "What is our Q3 revenue?"
        context = "=== Source: search_knowledge_base ===\n[1] Source: report.pdf (Page 2)\nQ3 revenue reached $15 million."
        generation = "According to [1] Source: report.pdf (Page 2), our Q3 revenue reached $15 million."

        report = self.evaluator.evaluate(query, context, generation)
        self.assertTrue(report.is_valid)
        self.assertTrue(report.is_sufficient)
        self.assertFalse(report.hallucination_detected)
        self.assertTrue(report.citations_valid)
        self.assertGreaterEqual(report.groundedness_score, 0.9)

    def test_hallucination_detection_on_insufficient_context(self):
        """Verify that inventing facts when context is insufficient triggers hallucination flag."""
        self.mock_llm.invoke.side_effect = [
            AIMessage(content='{"is_sufficient": false, "relevance_score": 0.1, "missing_information": "Severance package details"}'),
            AIMessage(content='{"groundedness_score": 0.2, "hallucination_detected": true, "citations_valid": false, "is_valid": false, "feedback": "Invented severance policy numbers."}'),
        ]

        query = "What is the severance package for executives?"
        context = "=== Source: search_knowledge_base ===\n[1] Source: handbook.pdf\nStandard office hours are 9am to 5pm."
        # The model made up a number not in context
        generation = "Executive severance packages include 24 months of full salary and equity vesting."

        report = self.evaluator.evaluate(query, context, generation)
        self.assertFalse(report.is_valid)
        self.assertTrue(report.hallucination_detected)
        self.assertFalse(report.is_sufficient)
        self.assertIn("does not contain sufficient information", report.feedback)

    def test_graceful_acknowledgment_of_insufficient_info(self):
        """Verify that when context is insufficient, explicitly stating it is considered grounded."""
        self.mock_llm.invoke.side_effect = [
            AIMessage(content='{"is_sufficient": false, "relevance_score": 0.0, "missing_information": "Severance policy"}'),
            AIMessage(content='{"groundedness_score": 1.0, "hallucination_detected": false, "citations_valid": true, "is_valid": true, "feedback": ""}'),
        ]

        query = "What is the severance package for executives?"
        context = "=== Source: search_knowledge_base ===\nNo relevant internal documents found."
        # Model correctly states the knowledge base does not have the information
        generation = "The internal knowledge base does not contain sufficient information regarding executive severance packages."

        report = self.evaluator.evaluate(query, context, generation)
        self.assertTrue(report.is_valid)
        self.assertFalse(report.hallucination_detected)


if __name__ == "__main__":
    unittest.main()

