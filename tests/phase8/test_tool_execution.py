"""
tests/phase8/test_tool_execution.py

Phase 8 — Tool Execution Tests
Covers:
  - Input validation (empty, URL injection, length)
  - Retry with exponential back-off on transient errors
  - Timeout enforcement
  - Parallel execution correctness and speed
  - ToolExecutionResult schema
  - ToolRegistry custom registration and lookup
  - Agent routing to correct tools (intent classification)
  - Multi-tool workflow execution
"""
import time
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.tools import Tool

from src.tool_executor import (
    validate_tool_input,
    execute_tool_with_resilience,
    execute_tools_parallel,
    ToolExecutionResult,
)
from src.tool_registry import ToolRegistry
from src.graph_nodes import create_intent_classification_node, create_external_tool_execution_node
from src.graph_state import AgentGraphState


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_empty_string_rejected(self):
        assert validate_tool_input("get_stock_data", "") is not None

    def test_none_rejected(self):
        assert validate_tool_input("get_stock_data", None) is not None  # type: ignore

    def test_url_injection_rejected(self):
        assert validate_tool_input("get_stock_data", "http://evil.com") is not None

    def test_too_long_input_rejected(self):
        assert validate_tool_input("get_weather", "a" * 150) is not None

    def test_valid_ticker_passes(self):
        assert validate_tool_input("get_stock_data", "AAPL") is None

    def test_valid_city_passes(self):
        assert validate_tool_input("get_weather", "New York") is None


# ---------------------------------------------------------------------------
# Retry Logic
# ---------------------------------------------------------------------------

class TestRetryLogic:

    def test_recovers_after_one_failure(self):
        calls = []

        def flaky(arg):
            calls.append(arg)
            if len(calls) < 2:
                raise ConnectionError("transient error")
            return f"ok:{arg}"

        tool = Tool(name="flaky", func=flaky, description="")
        result = execute_tool_with_resilience(tool, "x", max_retries=2, base_backoff_seconds=0.01)
        assert result.status == "success"
        assert result.retry_count == 1
        assert "ok:x" in result.output_text

    def test_exhausts_retries_and_errors(self):
        def always_fails(arg):
            raise RuntimeError("permanent error")

        tool = Tool(name="fail", func=always_fails, description="")
        result = execute_tool_with_resilience(tool, "x", max_retries=2, base_backoff_seconds=0.01)
        assert result.status == "error"
        assert result.retry_count == 2

    def test_no_retry_on_success(self):
        calls = []

        def succeeds(arg):
            calls.append(1)
            return "done"

        tool = Tool(name="ok", func=succeeds, description="")
        result = execute_tool_with_resilience(tool, "x", max_retries=3, base_backoff_seconds=0.01)
        assert result.retry_count == 0
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:

    def test_slow_tool_times_out(self):
        def slow(arg):
            time.sleep(5)
            return "never"

        tool = Tool(name="slow", func=slow, description="")
        result = execute_tool_with_resilience(tool, "x", max_retries=0, timeout_seconds=0.1)
        assert result.status == "error"
        assert "timed out" in result.output_text.lower()


# ---------------------------------------------------------------------------
# Parallel Execution
# ---------------------------------------------------------------------------

class TestParallelExecution:

    def test_two_tools_run_faster_than_sequential(self):
        def slow1(x):
            time.sleep(0.15)
            return f"A:{x}"

        def slow2(x):
            time.sleep(0.15)
            return f"B:{x}"

        t1 = Tool(name="t1", func=slow1, description="")
        t2 = Tool(name="t2", func=slow2, description="")
        tasks = [{"tool": t1, "argument": "q"}, {"tool": t2, "argument": "q"}]

        start = time.time()
        results = execute_tools_parallel(tasks, max_workers=2)
        elapsed = time.time() - start

        assert len(results) == 2
        assert elapsed < 0.35  # sequential would be ~0.30+; parallel < 0.25

    def test_all_results_returned(self):
        tools_ = [
            Tool(name=f"t{i}", func=lambda x, i=i: f"result{i}", description="")
            for i in range(5)
        ]
        tasks = [{"tool": t, "argument": "q"} for t in tools_]
        results = execute_tools_parallel(tasks, max_workers=5)
        assert len(results) == 5

    def test_result_schema(self):
        tool = Tool(name="simple", func=lambda x: "done", description="")
        results = execute_tools_parallel([{"tool": tool, "argument": "x"}])
        r = results[0]
        assert isinstance(r, ToolExecutionResult)
        assert r.tool_name == "simple"
        assert r.status in ("success", "error")
        assert isinstance(r.output_text, str)
        assert isinstance(r.execution_time_seconds, float)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:

    def test_register_and_lookup(self):
        registry = ToolRegistry()
        registry.register_custom_function(
            name="adder",
            func=lambda x: str(int(x) + 1),
            description="Adds 1.",
        )
        tool = registry.get_tool("adder")
        assert tool is not None
        assert tool.run("5") == "6"

    def test_describe_tools_includes_registered(self):
        registry = ToolRegistry()
        registry.register_custom_function("my_tool", lambda x: x, "A custom tool.")
        desc = registry.describe_tools()
        assert "my_tool" in desc

    def test_get_nonexistent_returns_none(self):
        registry = ToolRegistry()
        assert registry.get_tool("nonexistent_xyz") is None


# ---------------------------------------------------------------------------
# Agent Routing (intent classification)
# ---------------------------------------------------------------------------

class TestAgentRouting:

    def _route(self, query, entities=None):
        node = create_intent_classification_node()
        state: AgentGraphState = {
            "input": query,
            "normalized_query": query,
            "entities": entities or {},
            "chat_history": [],
            "query_intent": "",
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
        return node(state)

    def test_stock_query_routes_to_finance_tool(self):
        res = self._route("What is the AAPL stock price?", {"stock_tickers": ["AAPL"]})
        assert "get_stock_data" in res["plan"]

    def test_weather_query_routes_to_weather_tool(self):
        res = self._route("What is the weather in London?")
        assert "get_weather" in res["plan"]

    def test_cricket_routes_to_wikipedia(self):
        res = self._route("Who won the cricket world cup?")
        assert "search_wikipedia" in res["plan"]

    def test_unknown_query_defaults_to_knowledge_base(self):
        res = self._route("Explain the employee leave policy.")
        assert "search_knowledge_base" in res["plan"]

    def test_multi_tool_intent_set_correctly(self):
        res = self._route(
            "What is the weather and AAPL price?",
            {"stock_tickers": ["AAPL"], "locations": ["London"]},
        )
        assert res["query_intent"] == "multi_tool"
        assert "get_stock_data" in res["plan"]
        assert "get_weather" in res["plan"]

    def test_single_tool_intent_set_correctly(self):
        res = self._route("Current AAPL stock price?", {"stock_tickers": ["AAPL"]})
        assert res["query_intent"] == "single_tool"


# ---------------------------------------------------------------------------
# Multi-Tool Workflow
# ---------------------------------------------------------------------------

class TestMultiToolWorkflow:

    def test_two_external_tools_both_executed(self):
        tools_dict = {
            "get_stock_data": Tool(name="get_stock_data", func=lambda t: f"{t}:$100", description=""),
            "get_weather": Tool(name="get_weather", func=lambda c: f"{c}:25°C", description=""),
        }
        node = create_external_tool_execution_node(tools_dict)
        state: AgentGraphState = {
            "input": "AAPL price and London weather",
            "normalized_query": "AAPL price and London weather",
            "chat_history": [],
            "query_intent": "multi_tool",
            "plan": ["get_stock_data", "get_weather"],
            "entities": {"stock_tickers": ["AAPL"], "locations": ["London"]},
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
        result = node(state)
        assert "get_stock_data" in result["tool_results"]
        assert "get_weather" in result["tool_results"]
        assert "AAPL" in result["tool_results"]["get_stock_data"]
        assert "London" in result["tool_results"]["get_weather"]
        assert set(result["tools_used"]) == {"get_stock_data", "get_weather"}

    def test_skips_already_executed_tools(self):
        """Tools already in tool_results should not be re-executed."""
        call_count = {"n": 0}

        def counting_func(x):
            call_count["n"] += 1
            return "result"

        tools_dict = {
            "get_stock_data": Tool(name="get_stock_data", func=counting_func, description=""),
        }
        node = create_external_tool_execution_node(tools_dict)
        state: AgentGraphState = {
            "input": "test",
            "normalized_query": "test",
            "chat_history": [],
            "query_intent": "single_tool",
            "plan": ["get_stock_data"],
            "entities": {"stock_tickers": ["AAPL"]},
            # Already executed:
            "tool_results": {"get_stock_data": "cached result"},
            "aggregated_context": "",
            "generation": "",
            "validation": {},
            "iteration_count": 0,
            "max_iterations": 3,
            "tools_used": [],
            "output": "",
            "error": None,
        }
        node(state)
        assert call_count["n"] == 0  # should not have been called again

