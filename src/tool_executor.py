"""
src/tool_executor.py

Advanced Tool Execution Engine:
1. Structured tool output schema (ToolExecutionResult).
2. Tool execution validation (input parameter sanity checks).
3. Configurable tool-specific timeouts.
4. Retry mechanisms for transient network/API failures with exponential backoff.
5. Concurrent/parallel execution for independent tool invocations via ThreadPoolExecutor.
6. Clear, actionable error messaging.
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)


class ToolExecutionResult(BaseModel):
    """
    Standardized, structured return object for any tool execution.
    """
    tool_name: str = Field(description="Name of the tool executed.")
    status: str = Field(description="'success' or 'error'.")
    input_argument: Any = Field(description="Argument passed to the tool.")
    output_text: str = Field(description="Processed output text or human-readable error.")
    raw_data: Optional[Any] = Field(default=None, description="Optional raw JSON or dictionary payload.")
    execution_time_seconds: float = Field(default=0.0, description="Latency of execution in seconds.")
    retry_count: int = Field(default=0, description="Number of retries attempted before result.")
    error_message: Optional[str] = Field(default=None, description="Specific error details if failed.")


# Tool-specific timeouts (in seconds)
TOOL_TIMEOUTS: Dict[str, float] = {
    "get_stock_data": 8.0,
    "get_weather": 6.0,
    "search_wikipedia": 15.0,
    "web_search": 10.0,
    "search_knowledge_base": 8.0,
}


def validate_tool_input(tool_name: str, argument: Any) -> Optional[str]:
    """
    Validates input argument sanity before dispatching to tool.
    Returns None if valid, or an error explanation string if invalid.
    """
    if argument is None:
        return f"Tool '{tool_name}' requires an argument but received None."

    arg_str = str(argument).strip()
    if not arg_str:
        return f"Tool '{tool_name}' received an empty argument string."

    if tool_name == "get_stock_data":
        # Ticker should not exceed 15 chars or contain URL patterns
        if len(arg_str) > 20 or "http" in arg_str.lower():
            return f"Invalid ticker symbol '{arg_str}'. Expected a company name or ticker (e.g., AAPL)."

    elif tool_name == "get_weather":
        if len(arg_str) > 60 or "http" in arg_str.lower():
            return f"Invalid city name '{arg_str}'. Expected a city name (e.g., London)."

    return None


def execute_tool_with_resilience(
    tool: Tool,
    argument: Any,
    max_retries: int = 1,
    base_backoff_seconds: float = 0.5,
    timeout_seconds: Optional[float] = None,
) -> ToolExecutionResult:
    """
    Executes a tool with input validation, timeout capping, retry on transient failures,
    and returns a structured ToolExecutionResult.
    """
    tool_name = tool.name
    timeout = timeout_seconds or TOOL_TIMEOUTS.get(tool_name, 10.0)

    # 1. Input Validation
    validation_err = validate_tool_input(tool_name, argument)
    if validation_err:
        return ToolExecutionResult(
            tool_name=tool_name,
            status="error",
            input_argument=argument,
            output_text=f"Validation Error: {validation_err}",
            error_message=validation_err,
        )

    start_time = time.time()
    last_exception = None
    retries_done = 0

    for attempt in range(max_retries + 1):
        try:
            # We run the tool invocation
            def _call():
                return tool.run(argument)

            # Enforce timeout via ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as single_exec:
                future = single_exec.submit(_call)
                raw_result = future.result(timeout=timeout)

            elapsed = round(time.time() - start_time, 3)
            return ToolExecutionResult(
                tool_name=tool_name,
                status="success",
                input_argument=argument,
                output_text=str(raw_result),
                raw_data=raw_result if isinstance(raw_result, dict) else None,
                execution_time_seconds=elapsed,
                retry_count=attempt,
                error_message=None,
            )

        except TimeoutError:
            last_exception = f"Tool execution timed out after {timeout} seconds."
            logger.warning("Tool '%s' timed out (attempt %d/%d)", tool_name, attempt + 1, max_retries + 1)
        except Exception as exc:
            last_exception = str(exc)
            logger.warning("Tool '%s' failed on attempt %d: %s", tool_name, attempt + 1, exc)

        retries_done = attempt
        if attempt < max_retries:
            time.sleep(base_backoff_seconds * (2 ** attempt))

    elapsed = round(time.time() - start_time, 3)
    error_msg = f"Failed after {retries_done + 1} attempt(s): {last_exception}"
    return ToolExecutionResult(
        tool_name=tool_name,
        status="error",
        input_argument=argument,
        output_text=f"Tool '{tool_name}' encountered an error: {error_msg}",
        execution_time_seconds=elapsed,
        retry_count=retries_done,
        error_message=error_msg,
    )


def execute_tools_parallel(
    tool_tasks: List[Dict[str, Any]],
    max_workers: int = 4,
) -> List[ToolExecutionResult]:
    """
    Executes multiple independent tool calls in parallel using a thread pool.

    Each item in tool_tasks is a dict:
        {"tool": Tool, "argument": Any, "timeout": Optional[float]}

    Returns results in order of completion or original order.
    """
    if not tool_tasks:
        return []

    # If only 1 task, execute directly without thread pool overhead
    if len(tool_tasks) == 1:
        task = tool_tasks[0]
        return [
            execute_tool_with_resilience(
                tool=task["tool"],
                argument=task.get("argument", ""),
                timeout_seconds=task.get("timeout"),
            )
        ]

    results: List[Optional[ToolExecutionResult]] = [None] * len(tool_tasks)
    with ThreadPoolExecutor(max_workers=min(len(tool_tasks), max_workers)) as executor:
        future_to_index = {}
        for idx, task in enumerate(tool_tasks):
            fut = executor.submit(
                execute_tool_with_resilience,
                tool=task["tool"],
                argument=task.get("argument", ""),
                timeout_seconds=task.get("timeout"),
            )
            future_to_index[fut] = idx

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                res = future.result()
            except Exception as exc:
                task = tool_tasks[idx]
                res = ToolExecutionResult(
                    tool_name=task["tool"].name,
                    status="error",
                    input_argument=task.get("argument"),
                    output_text=f"Fatal thread error: {exc}",
                    error_message=str(exc),
                )
            results[idx] = res

    return [r for r in results if r is not None]

