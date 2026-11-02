"""
tests/test_session_memory.py

Unit and integration tests for Phase 5 Conversation Memory:
1. Thread-level checkpoint persistence
2. Context-aware multi-turn history tracking
3. Session isolation (verifying no cross-talk between Session A and Session B)
4. Session reset / starting a new conversation
"""

import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver

from src.agent import create_agent, run_agent_query
from src.session_manager import SessionManager


def mock_llm_responder(messages):
    """
    Deterministic response generator that inspects system prompt or content
    to return appropriate JSON or textual outputs for all nodes.
    """
    sys_content = messages[0].content if messages else ""
    user_content = messages[1].content if len(messages) > 1 else ""

    if "query analysis" in sys_content.lower():
        return AIMessage(content='{"normalized_query": "What is the query?", "entities": {"stock_tickers": ["AAPL"]}}')
    elif "intent classification" in sys_content.lower():
        if "what is apple stock" in user_content.lower():
            return AIMessage(content='{"intent": "single_tool", "plan": ["get_stock_data"]}')
        return AIMessage(content='{"intent": "direct", "plan": []}')
    elif "retrieval relevance" in sys_content.lower():
        return AIMessage(content='{"is_sufficient": true, "relevance_score": 1.0, "missing_information": ""}')
    elif "groundedness" in sys_content.lower():
        return AIMessage(content='{"groundedness_score": 1.0, "hallucination_detected": false, "citations_valid": true, "is_valid": true, "feedback": ""}')
    elif "quality-assurance" in sys_content.lower():
        return AIMessage(content='{"is_valid": true, "feedback": "", "confidence": 1.0}')
    else:
        # Answer generation
        if "blue" in user_content.lower():
            return AIMessage(content="Got it, your favorite color is Blue.")
        elif "favorite color" in user_content.lower():
            return AIMessage(content="I do not know your favorite color.")
        elif "ticker" in user_content.lower():
            return AIMessage(content="The ticker symbol mentioned is AAPL.")
        elif "apple" in user_content.lower():
            return AIMessage(content="Apple stock is $150.")
        return AIMessage(content="Answer generated successfully.")


class TestConversationMemory(unittest.TestCase):

    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_llm.invoke.side_effect = mock_llm_responder

        self.dummy_tools = [
            Tool(name="get_stock_data", func=lambda x: f"Stock data for {x}: $150", description="Stock tool"),
        ]
        self.checkpointer = MemorySaver()
        self.agent = create_agent(
            tools=self.dummy_tools,
            llm=self.mock_llm,
            memory=True,
            checkpointer=self.checkpointer,
            verbose=False,
        )
        self.session_manager = SessionManager(checkpointer=self.checkpointer)

    def test_multi_turn_history_persistence(self):
        """Verify that multiple turns in the same session accumulate into chat_history."""
        sid = self.session_manager.create_session("session_1")

        # Turn 1
        res1 = run_agent_query("What is Apple stock price?", self.agent, session_id=sid)
        self.assertEqual(res1["output"], "Apple stock is $150.")

        # Turn 2 (follow-up)
        res2 = run_agent_query("What was its ticker?", self.agent, session_id=sid)
        self.assertEqual(res2["output"], "The ticker symbol mentioned is AAPL.")

        # Check recorded history
        history = self.session_manager.get_session_history(self.agent, sid)
        self.assertEqual(len(history), 4)

    def test_session_isolation(self):
        """Verify that Session A and Session B have completely independent histories."""
        sid_a = self.session_manager.create_session("session_A")
        sid_b = self.session_manager.create_session("session_B")

        run_agent_query("My favorite color is Blue.", self.agent, session_id=sid_a)
        run_agent_query("What is my favorite color?", self.agent, session_id=sid_b)

        history_a = self.session_manager.get_session_history(self.agent, sid_a)
        history_b = self.session_manager.get_session_history(self.agent, sid_b)

        self.assertEqual(len(history_a), 2)
        self.assertEqual(len(history_b), 2)
        self.assertIn("Blue", history_a[0]["content"])
        self.assertNotIn("Blue", history_b[0]["content"])

    def test_reset_session(self):
        """Verify that resetting a session creates a clean new thread with zero history."""
        sid_initial = self.session_manager.create_session("session_to_reset")

        run_agent_query("Remember the secret password is alpha123", self.agent, session_id=sid_initial)
        initial_history = self.session_manager.get_session_history(self.agent, sid_initial)
        self.assertEqual(len(initial_history), 2)

        new_sid = self.session_manager.reset_session(sid_initial)
        self.assertNotEqual(new_sid, sid_initial)

        new_history = self.session_manager.get_session_history(self.agent, new_sid)
        self.assertEqual(len(new_history), 0)


if __name__ == "__main__":
    unittest.main()
