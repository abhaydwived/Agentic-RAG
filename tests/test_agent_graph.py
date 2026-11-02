"""
tests/test_agent_graph.py

Integration and unit tests for the LangGraph Agentic RAG workflow:
- Verifies graph compilation and node graph structure
- Tests query understanding, intent classification, and routing
- Tests multi-tool planning and conditional edges
- Tests execution without infinite loops
"""

import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.tools import Tool
from langgraph.graph import END

from src.agent import create_agent, route_after_intent, route_after_docs, route_after_validation
from src.graph_state import AgentGraphState


class TestAgentGraph(unittest.TestCase):

    def setUp(self):
        # Create a mock LLM for testing node logic deterministically
        self.mock_llm = MagicMock()

        # Dummy tools
        self.dummy_tools = [
            Tool(name="get_stock_data", func=lambda x: f"Stock data for {x}: $150", description="Stock tool"),
            Tool(name="get_weather", func=lambda x: f"Weather in {x}: 20°C sunny", description="Weather tool"),
            Tool(name="search_wikipedia", func=lambda x: f"Wikipedia article on {x}", description="Wikipedia tool"),
            Tool(name="web_search", func=lambda x: f"Web search results for {x}", description="Web search tool"),
        ]

    def test_graph_compilation(self):
        """Verify that the LangGraph StateGraph compiles successfully."""
        app = create_agent(
            tools=self.dummy_tools,
            llm=self.mock_llm,
            vector_store=None,
            memory=True,
            verbose=False,
        )
        self.assertIsNotNone(app)
        self.assertTrue(hasattr(app, "invoke"))

    def test_routing_after_intent(self):
        """Verify proper routing based on intent classification plan."""
        # 1. RAG planned -> document_retrieval
        state1: AgentGraphState = {
            "plan": ["search_knowledge_base"],
            "input": "test",
            "chat_history": [],
            "normalized_query": "test",
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
        self.assertEqual(route_after_intent(state1), "document_retrieval")

        # 2. External tools planned -> external_tools
        state2 = dict(state1)
        state2["plan"] = ["get_stock_data"]
        self.assertEqual(route_after_intent(state2), "external_tools")

        # 3. Direct response -> answer_generation
        state3 = dict(state1)
        state3["plan"] = []
        self.assertEqual(route_after_intent(state3), "answer_generation")

    def test_routing_after_docs(self):
        """Verify multi-tool vs single-tool routing after document retrieval."""
        # Multi-tool: plan has both RAG and finance
        state_multi: AgentGraphState = {
            "plan": ["search_knowledge_base", "get_stock_data"],
            "input": "compare valuation and price",
            "chat_history": [],
            "normalized_query": "",
            "query_intent": "multi_tool",
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
        self.assertEqual(route_after_docs(state_multi), "external_tools")

        # Single RAG tool: plan only has search_knowledge_base
        state_single = dict(state_multi)
        state_single["plan"] = ["search_knowledge_base"]
        self.assertEqual(route_after_docs(state_single), "context_aggregation")

    def test_routing_after_validation(self):
        """Verify termination condition and loop cap."""
        # Valid response -> END
        state_valid: AgentGraphState = {
            "validation": {"is_valid": True, "feedback": ""},
            "iteration_count": 1,
            "max_iterations": 3,
            "input": "",
            "chat_history": [],
            "normalized_query": "",
            "query_intent": "",
            "entities": {},
            "plan": [],
            "tool_results": {},
            "aggregated_context": "",
            "generation": "Done",
            "tools_used": [],
            "output": "Done",
            "error": None,
        }
        self.assertEqual(route_after_validation(state_valid), END)

        # Invalid response with iterations remaining -> answer_generation for revision
        state_invalid = dict(state_valid)
        state_invalid["validation"] = {"is_valid": False, "feedback": "Missing detail"}
        state_invalid["iteration_count"] = 1
        self.assertEqual(route_after_validation(state_invalid), "answer_generation")

        # Invalid response but reached max iterations -> END
        state_maxed = dict(state_invalid)
        state_maxed["iteration_count"] = 3
        self.assertEqual(route_after_validation(state_maxed), END)

    def test_end_to_end_mocked_invocation(self):
        """Test full graph execution with mock LLM returns to ensure no loop crashes."""
        # Answer generation invokes the LLM once in the single-call architecture
        self.mock_llm.invoke.return_value = AIMessage(
            content='AAPL is currently trading at $150 based on market data.'
        )

        app = create_agent(
            tools=self.dummy_tools,
            llm=self.mock_llm,
            vector_store=None,
            memory=True,
            verbose=False,
        )

        initial_state: AgentGraphState = {
            "input": "What is AAPL price?",
            "chat_history": [],
            "normalized_query": "",
            "query_intent": "direct",
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

        result = app.invoke(initial_state, config={"configurable": {"thread_id": "test_thread"}})

        self.assertIn("AAPL is currently trading at $150", result.get("output"))
        self.assertIn("get_stock_data", result.get("tools_used"))
        self.assertEqual(result.get("iteration_count"), 1)


if __name__ == "__main__":
    unittest.main()

