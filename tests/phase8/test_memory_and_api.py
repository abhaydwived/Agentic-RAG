"""
tests/phase8/test_memory_and_api.py

Phase 8 — Memory Isolation, API Endpoints, and Error Handling Tests
Covers:
  - Multi-turn conversation history persistence
  - Session isolation (no cross-talk between Session A and Session B)
  - Session reset / thread deletion
  - FastAPI handler logic (health_check, list_sessions, reset_session_endpoint, upload_document, chat_endpoint)
  - Error handling (unsupported upload file types, malformed chat payloads)
"""
import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from langchain_core.messages import AIMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver

from src.server import (
    health_check,
    list_sessions,
    reset_session_endpoint,
    chat_endpoint,
    upload_document,
    ChatRequest,
)
from src.session_manager import SessionManager
from src.agent import create_agent, run_agent_query


# ---------------------------------------------------------------------------
# 1. Memory Isolation & Session Management
# ---------------------------------------------------------------------------

class TestMemoryIsolation:

    def setup_method(self):
        self.mock_llm = MagicMock()
        # Mock answers for session turns
        self.mock_llm.invoke.side_effect = lambda msgs: AIMessage(
            content="Answer: " + (msgs[-1].content if hasattr(msgs[-1], 'content') else str(msgs[-1]))
        )
        self.tools = [
            Tool(name="get_stock_data", func=lambda x: f"Stock {x}: $150", description=""),
        ]
        self.checkpointer = MemorySaver()
        self.agent = create_agent(
            tools=self.tools,
            llm=self.mock_llm,
            memory=True,
            checkpointer=self.checkpointer,
            verbose=False,
        )
        self.session_manager = SessionManager(checkpointer=self.checkpointer)

    def test_multi_turn_history_persistence(self):
        sid = self.session_manager.create_session("sess_persistent")
        run_agent_query("My name is Alice.", self.agent, session_id=sid)
        run_agent_query("What is my name?", self.agent, session_id=sid)

        history = self.session_manager.get_session_history(self.agent, sid)
        assert len(history) >= 4
        contents = [h["content"] for h in history]
        assert any("Alice" in c for c in contents)

    def test_strict_session_isolation(self):
        """User data in Session A must NEVER leak into Session B."""
        sid_a = self.session_manager.create_session("session_alpha")
        sid_b = self.session_manager.create_session("session_beta")

        run_agent_query("Confidential secret is code_xyz.", self.agent, session_id=sid_a)
        run_agent_query("What is the secret?", self.agent, session_id=sid_b)

        history_a = self.session_manager.get_session_history(self.agent, sid_a)
        history_b = self.session_manager.get_session_history(self.agent, sid_b)

        assert any("code_xyz" in h["content"] for h in history_a)
        assert not any("code_xyz" in h["content"] for h in history_b)

    def test_reset_session_creates_clean_slate(self):
        sid = self.session_manager.create_session("session_to_wipe")
        run_agent_query("Remember number 42.", self.agent, session_id=sid)
        assert len(self.session_manager.get_session_history(self.agent, sid)) >= 2

        new_sid = self.session_manager.reset_session(sid)
        assert new_sid != sid
        assert len(self.session_manager.get_session_history(self.agent, new_sid)) == 0


# ---------------------------------------------------------------------------
# 2. API Endpoints & Error Handling (Direct Handlers)
# ---------------------------------------------------------------------------

class TestServerEndpoints:

    def test_health_handler(self):
        data = health_check()
        assert data["status"] == "healthy"
        assert "tools_registered" in data
        assert "search_knowledge_base" in data["tools_registered"]

    def test_list_and_delete_sessions(self):
        data = list_sessions()
        assert "active_sessions" in data
        assert "count" in data

        res = reset_session_endpoint("test_dummy_session")
        assert res["status"] == "cleared"
        assert res["old_session_id"] == "test_dummy_session"
        assert "new_session_id" in res

    def test_upload_invalid_file_extension(self):
        """Rejects executable or non-doc files (.sh, .exe, .py)."""
        mock_file = MagicMock()
        mock_file.filename = "exploit.sh"
        with pytest.raises(HTTPException) as ctx:
            upload_document(mock_file)
        assert ctx.value.status_code == 400
        assert "Unsupported file type" in ctx.value.detail

    @patch("src.server.run_agent_query")
    @patch("src.server.get_agent_instance")
    def test_chat_endpoint_success(self, mock_agent_inst, mock_query):
        mock_agent_inst.return_value = MagicMock()
        mock_query.return_value = {
            "output": "According to handbook, remote work is 3 days/week.",
            "tools_used": ["search_knowledge_base"],
            "tool_results": {
                "search_knowledge_base": "[1] Source: handbook.txt: Remote work up to 3 days."
            },
            "validation": {"is_valid": True},
            "iteration_count": 1,
        }

        req = ChatRequest(query="What is remote work policy?", session_id="test_sess")
        resp = chat_endpoint(req)
        assert "remote work is 3 days/week" in resp.answer
        assert "search_knowledge_base" in resp.tools_used
        assert resp.retrieved_sources is not None
        assert resp.session_id == "test_sess"
