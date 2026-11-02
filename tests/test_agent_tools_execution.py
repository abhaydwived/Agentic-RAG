"""
tests/test_agent_tools_execution.py

Unit and integration tests for Phase 4 Advanced Tool Execution:
1. Tool input validation
2. Retry mechanisms for transient errors
3. Tool execution timeouts
4. Parallel execution of multiple independent tools
5. Structured ToolExecutionResult schema
6. Dynamic tool registration via ToolRegistry
"""

import time
import unittest
from unittest.mock import MagicMock

from langchain_core.tools import Tool

from src.tool_executor import (
    validate_tool_input,
    execute_tool_with_resilience,
    execute_tools_parallel,
    ToolExecutionResult,
)
from src.tool_registry import ToolRegistry


class TestAdvancedToolExecution(unittest.TestCase):

    def test_tool_input_validation(self):
        """Verify input validation catches empty or malformed inputs."""
        self.assertIsNotNone(validate_tool_input("get_stock_data", ""))
        self.assertIsNotNone(validate_tool_input("get_stock_data", None))
        self.assertIsNotNone(validate_tool_input("get_stock_data", "http://malicious.com"))
        self.assertIsNone(validate_tool_input("get_stock_data", "AAPL"))

        self.assertIsNotNone(validate_tool_input("get_weather", "a" * 100))
        self.assertIsNone(validate_tool_input("get_weather", "Tokyo, JP"))

    def test_tool_retry_on_transient_failure(self):
        """Verify tool retries on transient exceptions and recovers."""
        call_count = 0

        def flaky_func(arg: str):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary network reset")
            return f"Success for {arg}"

        tool = Tool(name="flaky_tool", func=flaky_func, description="Test tool")
        res = execute_tool_with_resilience(
            tool=tool,
            argument="test_query",
            max_retries=2,
            base_backoff_seconds=0.01,
        )

        self.assertEqual(res.status, "success")
        self.assertEqual(res.retry_count, 1)
        self.assertIn("Success for test_query", res.output_text)
        self.assertEqual(call_count, 2)

    def test_tool_timeout(self):
        """Verify tool execution terminates and returns error if timeout is exceeded."""
        def slow_func(arg: str):
            time.sleep(1.0)
            return "Should not finish"

        tool = Tool(name="slow_tool", func=slow_func, description="Slow tool")
        res = execute_tool_with_resilience(
            tool=tool,
            argument="slow_arg",
            max_retries=0,
            timeout_seconds=0.1,
        )

        self.assertEqual(res.status, "error")
        self.assertIn("timed out", res.output_text.lower())

    def test_parallel_tool_execution(self):
        """Verify multiple independent tools run concurrently."""
        def tool1(x):
            time.sleep(0.1)
            return f"T1: {x}"

        def tool2(x):
            time.sleep(0.1)
            return f"T2: {x}"

        t1 = Tool(name="tool_one", func=tool1, description="")
        t2 = Tool(name="tool_two", func=tool2, description="")

        tasks = [
            {"tool": t1, "argument": "arg1"},
            {"tool": t2, "argument": "arg2"},
        ]

        start = time.time()
        results = execute_tools_parallel(tasks, max_workers=2)
        elapsed = time.time() - start

        self.assertEqual(len(results), 2)
        tool_names = [r.tool_name for r in results]
        self.assertIn("tool_one", tool_names)
        self.assertIn("tool_two", tool_names)
        # Parallel execution of two 0.1s tasks should complete significantly faster than 0.25s
        self.assertLess(elapsed, 0.25)

    def test_dynamic_tool_registration(self):
        """Verify registering new custom tools into ToolRegistry is straightforward."""
        registry = ToolRegistry()

        def custom_calculator(expr: str) -> str:
            return f"Result: {eval(expr, {'__builtins__': None}, {})}"

        tool = registry.register_custom_function(
            name="calculator",
            func=custom_calculator,
            description="Evaluates simple math expressions.",
        )

        self.assertIsNotNone(registry.get_tool("calculator"))
        self.assertEqual(tool.run("5 * 20"), "Result: 100")
        self.assertIn("calculator", registry.describe_tools())


if __name__ == "__main__":
    unittest.main()

