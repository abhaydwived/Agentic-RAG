"""
src/session_manager.py

LangGraph-native Session & Conversation Memory Management:
- Session-specific checkpointers (MemorySaver)
- Conversation history tracking with message sequence preservation
- Session isolation (no cross-contamination between different sessions)
- Ability to start new sessions, clear sessions, or switch active threads
- Context-aware retrieval of recent conversation turns
"""

import uuid
import logging
from typing import Any, Dict, List, Optional
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages session identifiers, thread configurations, and memory checkpointers.
    """

    def __init__(self, checkpointer: Optional[MemorySaver] = None):
        # Shared checkpointer indexes state snapshots by thread_id
        self.checkpointer = checkpointer or MemorySaver()
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Creates a new unique session identifier."""
        sid = session_id or str(uuid.uuid4())
        self._active_sessions[sid] = {
            "created_at": None,
            "turns_count": 0,
        }
        return sid

    def get_thread_config(self, session_id: str) -> Dict[str, Any]:
        """
        Builds the LangGraph runnable configuration with the thread_id.
        LangGraph uses configurable.thread_id to isolate checkpoints completely.
        """
        if session_id not in self._active_sessions:
            self.create_session(session_id)
        return {"configurable": {"thread_id": str(session_id)}}

    def get_session_history(self, app, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the persisted conversation turns for a given session.
        """
        config = self.get_thread_config(session_id)
        try:
            state = app.get_state(config)
            if state and state.values:
                return state.values.get("chat_history", [])
        except Exception as exc:
            logger.debug("Could not fetch state for session %s: %s", session_id, exc)
        return []

    def reset_session(self, session_id: str) -> str:
        """
        Resets/clears a session by generating a fresh thread identifier,
        guaranteeing zero state spillover.
        """
        new_sid = str(uuid.uuid4())
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        self._active_sessions[new_sid] = {"turns_count": 0}
        return new_sid

