"""
tests/test_all_nodes.py

Comprehensive tests verifying every individual node in src/graph_nodes.py:
1. query_understanding_node
2. intent_classification_node
3. document_retrieval_node
4. external_tool_execution_node
5. context_aggregation_node
6. answer_generation_node
7. response_validation_node
"""

import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.tools import Tool

from src.graph_nodes import (
    create_query_understanding_node,
    create_intent_classification_node,
    create_document_retrieval_node,
    create_external_tool_execution_node,
    create_context_aggregation_node,
    create_answer_generation_node,
    create_response_validation_node,
)
from src.graph_state import AgentGraphState


class TestAllNodes(unittest.TestCase):

    def setUp(self):
        self.mock_llm = MagicMock()

    def test_query_understanding_node(self):
        """Test extraction of normalized query and entities."""
        node = create_query_understanding_node(self.mock_llm)
        state: AgentGraphState = {
            "input": "What is the stock price of Tesla?",
            "chat_history": [],
            "normalized_query": "",
            "query_intent": "",
            "entities": {},
            "plan": [],
            "tool_results": {},
            "aggregated_context": "",
            "generation": "",
            "validation": {},
            "iteration_count": 0,
            "max_iterations": 3,
            "tools_used": [],
            "output": "",
            "error": None,
        }
        res = node(state)
        self.assertEqual(res["normalized_query"], "What is the stock price of Tesla?")
        self.assertEqual(res["entities"]["stock_tickers"], ["TSLA"])

    def test_intent_classification_node(self):
        """Test multi-tool planning and intent routing."""
        node = create_intent_classification_node(self.mock_llm)
        state: AgentGraphState = {
            "input": "What is the weather in Paris and AAPL stock price?",
            "chat_history": [],
            "normalized_query": "What is the weather in Paris and AAPL stock price?",
            "query_intent": "",
            "entities": {"stock_tickers": ["AAPL"], "locations": ["Paris"]},
            "plan": [],
            "tool_results": {},
            "aggregated_context": "",
            "generation": "",
            "validation": {},
            "iteration_count": 0,
            "max_iterations": 3,
            "tools_used": [],
            "output": "",
            "error": None,
        }
        res = node(state)
        self.assertEqual(res["query_intent"], "multi_tool")
        self.assertIn("search_knowledge_base" if "policy" in res["plan"] else "get_stock_data", res["plan"])
        self.assertIn("get_weather", res["plan"])

    def test_document_retrieval_node(self):
        """Test vector store document retrieval when planned."""
        mock_vector_store = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "Q3 financial internal revenue target is $10M."
        mock_vector_store.similarity_search.return_value = [mock_doc]

        node = create_document_retrieval_node(mock_vector_store)

        # 1. Planned
        state_planned: AgentGraphState = {
            "plan": ["search_knowledge_base"],
            "input": "What is revenue target?",
            "normalized_query": "What is revenue target?",
            "chat_history": [],
            "query_intent": "rag",
            "entities": {},
            "tool_results": {},
            "aggregated_context": "",
            "generation": "",
            "validation": {},
            "iteration_count": 0,
            "max_iterations": 3,
            "tools_used": [],
            "output": "",
            "error": None,
        }
        res = node(state_planned)
        self.assertIn("search_knowledge_base", res["tool_results"])
        self.assertIn("Q3", res["tool_results"]["search_knowledge_base"])

        # 2. Not planned
        state_not_planned = dict(state_planned)
        state_not_planned["plan"] = ["get_stock_data"]
        res_empty = node(state_not_planned)
        self.assertEqual(res_empty, {})

    def test_external_tool_execution_node(self):
        """Test external tool execution with entity dispatch."""
        tools_dict = {
            "get_stock_data": Tool(name="get_stock_data", func=lambda ticker: f"Stock price for {ticker}: $220", description=""),
            "get_weather": Tool(name="get_weather", func=lambda city: f"Weather in {city}: 18°C", description=""),
        }
        node = create_external_tool_execution_node(tools_dict)

        state: AgentGraphState = {
            "plan": ["get_stock_data", "get_weather"],
            "entities": {"stock_tickers": ["NVDA"], "locations": ["Tokyo"]},
            "tool_results": {},
            "input": "NVDA price and Tokyo weather",
            "chat_history": [],
            "normalized_query": "NVDA price and Tokyo weather",
            "query_intent": "multi_tool",
            "aggregated_context": "",
            "generation": "",
            "validation": {},
            "iteration_count": 0,
            "max_iterations": 3,
            "tools_used": [],
            "output": "",
            "error": None,
        }
        res = node(state)
        self.assertIn("get_stock_data", res["tool_results"])
        self.assertIn("NVDA: $220", res["tool_results"]["get_stock_data"])
        self.assertIn("Tokyo: 18°C", res["tool_results"]["get_weather"])
        self.assertEqual(res["tools_used"], ["get_stock_data", "get_weather"])

    def test_context_aggregation_node(self):
        """Test context aggregation and formatting."""
        node = create_context_aggregation_node()
        state: AgentGraphState = {
            "tool_results": {
                "search_knowledge_base": "[1] Company policy: remote work allowed.",
                "get_stock_data": "Stock price for AAPL: $180",
            },
            "input": "",
            "chat_history": [],
            "normalized_query": "",
            "query_intent": "",
            "entities": {},
            "plan": [],
            "aggregated_context": "",
            "generation": "",
            "validation": {},
            "iteration_count": 0,
            "max_iterations": 3,
            "tools_used": [],
            "output": "",
            "error": None,
        }
        res = node(state)
        self.assertIn("=== Source: search_knowledge_base ===", res["aggregated_context"])
        self.assertIn("=== Source: get_stock_data ===", res["aggregated_context"])

    def test_answer_generation_node(self):
        """Test answer generation with LLM."""
        self.mock_llm.invoke.return_value = AIMessage(content="According to the internal policy, remote work is allowed.")
        node = create_answer_generation_node(self.mock_llm)
        state: AgentGraphState = {
            "input": "Is remote work allowed?",
            "aggregated_context": "=== Source: search_knowledge_base ===\nRemote work policy: allowed.",
            "chat_history": [],
            "normalized_query": "",
            "query_intent": "",
            "entities": {},
            "plan": [],
            "tool_results": {},
            "generation": "",
            "validation": {},
            "iteration_count": 0,
            "max_iterations": 3,
            "tools_used": [],
            "output": "",
            "error": None,
        }
        res = node(state)
        self.assertEqual(res["generation"], "According to the internal policy, remote work is allowed.")

    def test_response_validation_node(self):
        """Test response validation and output assignment."""
        self.mock_llm.invoke.return_value = AIMessage(
            content='{"is_valid": true, "feedback": "", "confidence": 0.98}'
        )
        node = create_response_validation_node(self.mock_llm)
        state: AgentGraphState = {
            "input": "What is the policy?",
            "generation": "Remote work is allowed.",
            "aggregated_context": "Remote work allowed.",
            "iteration_count": 0,
            "max_iterations": 3,
            "chat_history": [],
            "normalized_query": "",
            "query_intent": "",
            "entities": {},
            "plan": [],
            "tool_results": {},
            "validation": {},
            "tools_used": [],
            "output": "",
            "error": None,
        }
        res = node(state)
        self.assertTrue(res["validation"]["is_valid"])
        self.assertEqual(res["iteration_count"], 1)
        self.assertEqual(res["output"], "Remote work is allowed.")


if __name__ == "__main__":
    unittest.main()

