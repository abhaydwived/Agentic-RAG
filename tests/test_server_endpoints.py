"""
tests/test_server_endpoints.py

Direct handler unit tests for FastAPI backend routes:
1. GET /health
2. GET /sessions
3. DELETE /sessions/{session_id}
4. POST /chat (mocked agent execution)
5. POST /upload (input validation)
"""

import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from src.server import (
    health_check,
    list_sessions,
    reset_session_endpoint,
    chat_endpoint,
    upload_document,
    ChatRequest,
)


class TestFastAPIServer(unittest.TestCase):

    def test_health_handler(self):
        """Verify health check logic returns tool registry and system metadata."""
        data = health_check()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("tools_registered", data)
        self.assertIn("get_stock_data", data["tools_registered"])

    def test_sessions_handlers(self):
        """Verify session listing and deletion endpoints."""
        data = list_sessions()
        self.assertIn("active_sessions", data)
        self.assertIn("count", data)

        res = reset_session_endpoint("session_xyz")
        self.assertEqual(res["status"], "cleared")
        self.assertEqual(res["old_session_id"], "session_xyz")
        self.assertIn("new_session_id", res)

    def test_upload_invalid_extension(self):
        """Verify upload handler rejects unsupported extensions."""
        mock_file = MagicMock()
        mock_file.filename = "malicious.sh"
        with self.assertRaises(HTTPException) as ctx:
            upload_document(mock_file)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Unsupported file type", ctx.exception.detail)

    @patch("src.server.get_agent_instance")
    @patch("src.server.run_agent_query")
    def test_chat_handler(self, mock_run_agent, mock_get_agent):
        """Verify chat handler processes request and separates knowledge base vs live data."""
        mock_get_agent.return_value = MagicMock()
        mock_run_agent.return_value = {
            "output": "AAPL is currently trading at $150.",
            "tools_used": ["get_stock_data"],
            "tool_results": {
                "get_stock_data": "Price: $150, PE: 30",
            },
            "validation": {"is_valid": True},
            "iteration_count": 1,
        }

        req = ChatRequest(query="What is AAPL price?", session_id="test_sess_1")
        resp = chat_endpoint(req)

        self.assertEqual(resp.answer, "AAPL is currently trading at $150.")
        self.assertIn("get_stock_data", resp.tools_used)
        self.assertIn("get_stock_data", resp.live_external_data)
        self.assertEqual(resp.session_id, "test_sess_1")


if __name__ == "__main__":
    unittest.main()
