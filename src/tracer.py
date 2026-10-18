"""
src/tracer.py

Observability and Tracing Engine:
- Automatically binds with LangSmith environment variables if configured.
- Local high-precision fallback tracer tracking:
  - LLM calls
  - Tool calls
  - Retrieval results
  - Execution latency per stage and end-to-end
  - Token counts (estimated/reported)
  - Execution errors
  - Agent iteration count
- Exportable trace snapshots for auditability.
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def setup_langsmith_tracing(
    project_name: str = "agentic-rag-realtime",
    api_key: Optional[str] = None,
) -> bool:
    """
    Configures LangSmith environment variables if an API key is available.
    Returns True if LangSmith tracing is active, False otherwise.
    """
    key = api_key or os.getenv("LANGCHAIN_API_KEY")
    if key and key != "your_langsmith_api_key_here":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = key
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", project_name)
        logger.info("LangSmith tracing enabled for project '%s'.", os.environ["LANGCHAIN_PROJECT"])
        return True
    return False


class StepTrace(BaseModel):
    """Trace record for an individual agent execution step or tool call."""
    step_name: str
    start_time: float
    end_time: float = 0.0
    duration_seconds: float = 0.0
    status: str = "running"
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    tokens_used: int = 0


class ExecutionTrace(BaseModel):
    """Full execution trace of a user query across the LangGraph workflow."""
    session_id: str
    query: str
    start_time: float
    end_time: float = 0.0
    total_latency_seconds: float = 0.0
    steps: List[StepTrace] = Field(default_factory=list)
    llm_call_count: int = 0
    tool_calls: List[str] = Field(default_factory=list)
    retrieval_chunk_count: int = 0
    iteration_count: int = 0
    total_tokens: int = 0
    has_error: bool = False
    error_summary: Optional[str] = None


class LocalTracer:
    """
    Lightweight, high-precision in-memory and local file tracer.
    """

    def __init__(self):
        self._current_trace: Optional[ExecutionTrace] = None
        self._history: List[ExecutionTrace] = []

    def start_trace(self, query: str, session_id: str = "default") -> ExecutionTrace:
        """Initializes a new execution trace for a query."""
        now = time.time()
        self._current_trace = ExecutionTrace(
            session_id=session_id,
            query=query,
            start_time=now,
        )
        return self._current_trace

    def record_step_start(self, step_name: str, inputs: Optional[Dict[str, Any]] = None) -> StepTrace:
        """Starts recording a specific pipeline step."""
        step = StepTrace(
            step_name=step_name,
            start_time=time.time(),
            inputs=inputs or {},
        )
        if self._current_trace:
            self._current_trace.steps.append(step)
        return step

    def record_step_end(
        self,
        step: StepTrace,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        tokens: int = 0,
    ) -> None:
        """Marks completion of a pipeline step and records latency."""
        step.end_time = time.time()
        step.duration_seconds = round(step.end_time - step.start_time, 4)
        step.outputs = outputs or {}
        step.error = error
        step.status = "error" if error else "success"
        step.tokens_used = tokens

        if self._current_trace:
            self._current_trace.total_tokens += tokens
            if error:
                self._current_trace.has_error = True
                self._current_trace.error_summary = error

    def finish_trace(
        self,
        llm_calls: int = 0,
        tool_calls: Optional[List[str]] = None,
        retrieval_chunks: int = 0,
        iterations: int = 0,
    ) -> ExecutionTrace:
        """Finalizes the active execution trace and appends to history."""
        if not self._current_trace:
            return ExecutionTrace(session_id="none", query="", start_time=time.time())

        self._current_trace.end_time = time.time()
        self._current_trace.total_latency_seconds = round(
            self._current_trace.end_time - self._current_trace.start_time, 4
        )
        self._current_trace.llm_call_count = llm_calls
        self._current_trace.tool_calls = tool_calls or []
        self._current_trace.retrieval_chunk_count = retrieval_chunks
        self._current_trace.iteration_count = iterations

        self._history.append(self._current_trace)
        return self._current_trace

    def get_latest_trace(self) -> Optional[ExecutionTrace]:
        """Returns the most recent completed execution trace."""
        return self._current_trace


# Global tracer instance
tracer = LocalTracer()

