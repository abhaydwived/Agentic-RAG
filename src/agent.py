"""
src/agent.py

LangGraph-based Agentic RAG workflow with:
- Checkpointed session memory (MemorySaver)
- Thread-level isolation preventing cross-session data leakage
- Query understanding node with coreference resolution
- Intent classification and routing node
- Document retrieval node
- External tool execution node with resilience and parallelism
- Context aggregation node
- Answer generation node
- Decoupled response validation node
- Conditional routing and termination with safety iteration limits
"""

from typing import Any, Dict, List, Optional
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS

from src.graph_state import AgentGraphState
from src.graph_nodes import (
    create_query_understanding_node,
    create_intent_classification_node,
    create_document_retrieval_node,
    create_external_tool_execution_node,
    create_context_aggregation_node,
    create_answer_generation_node,
    create_response_validation_node,
)
from src.session_manager import SessionManager

logger = logging.getLogger(__name__)


def route_after_intent(state: AgentGraphState) -> str:
    """
    Decides routing after intent classification:
    - If plan contains 'search_knowledge_base', route to 'document_retrieval'.
    - Else if plan contains other tools, route to 'external_tools'.
    - If 'direct' or no tools planned, route directly to 'answer_generation'.
    """
    plan = state.get("plan", [])
    if "search_knowledge_base" in plan:
        return "document_retrieval"
    elif plan:
        return "external_tools"
    return "answer_generation"


def route_after_docs(state: AgentGraphState) -> str:
    """
    Decides routing after document retrieval:
    - If remaining tools exist in the plan (multi-tool sequence), route to 'external_tools'.
    - Otherwise, proceed to 'context_aggregation'.
    """
    plan = state.get("plan", [])
    external_tools_in_plan = [t for t in plan if t != "search_knowledge_base"]
    if external_tools_in_plan:
        return "external_tools"
    return "context_aggregation"


def route_after_validation(state: AgentGraphState) -> str:
    """
    Conditional termination / revision routing:
    - If validation passed OR max iterations exceeded -> END.
    - If validation failed and iterations remain -> regenerate or refine via answer_generation.
    """
    validation = state.get("validation", {})
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if validation.get("is_valid", True) or iteration_count >= max_iterations:
        return END

    return "answer_generation"


def create_agent(
    tools: List[Tool],
    llm,
    vector_store: Optional[FAISS] = None,
    memory: bool = True,
    checkpointer: Optional[MemorySaver] = None,
    verbose: bool = True,
    max_iterations: int = 3,
):
    """
    Builds and compiles the LangGraph Agentic RAG StateGraph with checkpointer memory.

    Args:
        tools: List of LangChain Tool objects.
        llm: Instantiated chat model (e.g. ChatGoogleGenerativeAI or ChatOpenAI).
        vector_store: Optional FAISS vector store.
        memory: Whether to attach session checkpointer.
        checkpointer: Custom MemorySaver instance (optional).
        verbose: Verbose logging.
        max_iterations: Maximum loop iterations for validation & regeneration.

    Returns:
        Compiled LangGraph runnable workflow.
    """
    tools_by_name = {t.name: t for t in tools}

    if vector_store is None:
        for tool in tools:
            if hasattr(tool, "vector_store"):
                vector_store = getattr(tool, "vector_store")
                break

    # Build StateGraph
    builder = StateGraph(AgentGraphState)

    # 1. Add nodes
    builder.add_node("query_understanding", create_query_understanding_node(llm))
    builder.add_node("intent_classification", create_intent_classification_node(llm))
    builder.add_node("document_retrieval", create_document_retrieval_node(vector_store))
    builder.add_node("external_tools", create_external_tool_execution_node(tools_by_name))
    builder.add_node("context_aggregation", create_context_aggregation_node())
    builder.add_node("answer_generation", create_answer_generation_node(llm))
    builder.add_node("response_validation", create_response_validation_node(llm))

    # 2. Add edges
    builder.add_edge(START, "query_understanding")
    builder.add_edge("query_understanding", "intent_classification")

    builder.add_conditional_edges(
        "intent_classification",
        route_after_intent,
        {
            "document_retrieval": "document_retrieval",
            "external_tools": "external_tools",
            "answer_generation": "answer_generation",
        },
    )

    builder.add_conditional_edges(
        "document_retrieval",
        route_after_docs,
        {
            "external_tools": "external_tools",
            "context_aggregation": "context_aggregation",
        },
    )

    builder.add_edge("external_tools", "context_aggregation")
    builder.add_edge("context_aggregation", "answer_generation")
    builder.add_edge("answer_generation", "response_validation")

    builder.add_conditional_edges(
        "response_validation",
        route_after_validation,
        {
            END: END,
            "answer_generation": "answer_generation",
        },
    )

    # Attach MemorySaver checkpointer for thread-level state persistence
    saver = checkpointer if checkpointer is not None else (MemorySaver() if memory else None)
    return builder.compile(checkpointer=saver)


def run_agent_query(
    query: str,
    agent,
    session_id: Optional[str] = None,
    chat_history: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Submits a query to the LangGraph agent with session configuration.

    Args:
        query: User input query.
        agent: Compiled LangGraph workflow.
        session_id: Session identifier used for checkpoint thread isolation.
        chat_history: Optional prior conversation turns.

    Returns:
        State dict containing 'output', 'tools_used', 'generation', etc.
    """
    sid = session_id or "default_session"
    config = {"configurable": {"thread_id": sid}}

    # Fetch prior turns from checkpointer if available
    prior_history = []
    if agent.checkpointer:
        try:
            checkpoint = agent.get_state(config)
            if checkpoint and checkpoint.values:
                prior_history = checkpoint.values.get("chat_history", [])
        except Exception:
            pass

    if chat_history and not prior_history:
        prior_history = list(chat_history)

    initial_state: AgentGraphState = {
        "input": query,
        "chat_history": prior_history,
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

    try:
        final_state = agent.invoke(initial_state, config=config)
        output_ans = final_state.get("output") or final_state.get("generation", "No response generated.")
        final_state["output"] = output_ans

        # Append latest turn to chat_history in state
        new_turns = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": output_ans},
        ]
        if agent.checkpointer:
            updated_history = list(prior_history) + new_turns
            agent.update_state(config, {"chat_history": updated_history})

        return final_state
    except Exception as exc:
        logger.exception("Error executing agent graph: %s", exc)
        return {
            "output": f"Agent encountered an error: {exc}",
            "tools_used": [],
            "error": str(exc),
        }
