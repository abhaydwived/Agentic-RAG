"""
tests/conftest.py

Shared pytest fixtures used across all Phase 8 test modules.
All external API calls (Gemini, Wikipedia, Tavily, yfinance, OpenWeatherMap)
are mocked so tests run fully offline and without any GPU.
"""
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Environment: force HuggingFace offline so no network calls are made
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------

SAMPLE_TXT_CONTENT = (
    "Artificial Intelligence is transforming the modern world.\n\n"
    "Machine Learning models learn patterns from labeled training data.\n\n"
    "Deep neural networks use backpropagation to optimize weights."
)

SAMPLE_TXT_CONTENT_2 = (
    "Quantum computing leverages qubits and superposition.\n\n"
    "Shor's algorithm can factorize large integers exponentially faster.\n\n"
    "Quantum entanglement enables secure communication channels."
)


@pytest.fixture(scope="function")
def tmp_kb_dir():
    """Creates a temporary directory with sample .txt documents."""
    d = tempfile.mkdtemp()
    p1 = os.path.join(d, "ai_doc.txt")
    p2 = os.path.join(d, "quantum_doc.txt")
    with open(p1, "w", encoding="utf-8") as f:
        f.write(SAMPLE_TXT_CONTENT)
    with open(p2, "w", encoding="utf-8") as f:
        f.write(SAMPLE_TXT_CONTENT_2)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_docs():
    """Returns a list of pre-built Document objects."""
    return [
        Document(
            page_content="Deep neural networks require backpropagation to train.",
            metadata={"document_name": "ai.txt", "page": 1, "source": "ai.txt"},
        ),
        Document(
            page_content="Quantum computing leverages superposition of qubit states.",
            metadata={"document_name": "quantum.txt", "page": 1, "source": "quantum.txt"},
        ),
        Document(
            page_content="Photosynthesis converts solar energy into chemical energy in plants.",
            metadata={"document_name": "bio.txt", "page": 1, "source": "bio.txt"},
        ),
        Document(
            page_content="Employee remote work policy allows up to 3 days per week from home.",
            metadata={"document_name": "handbook.txt", "page": 1, "source": "handbook.txt"},
        ),
    ]


@pytest.fixture(scope="function")
def mock_llm():
    """A MagicMock LLM whose .invoke() returns a configurable AIMessage."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Mocked LLM answer.")
    return llm


@pytest.fixture(scope="function")
def dummy_tools():
    """Minimal tools list for agent tests — no real API calls."""
    return [
        Tool(
            name="get_stock_data",
            func=lambda ticker: f"Stock price for {ticker}: $200",
            description="Get real-time stock data.",
        ),
        Tool(
            name="get_weather",
            func=lambda city: f"Weather in {city}: 25°C, sunny",
            description="Get current weather.",
        ),
        Tool(
            name="search_wikipedia",
            func=lambda q: f"Wikipedia summary for: {q}",
            description="Search Wikipedia.",
        ),
        Tool(
            name="web_search",
            func=lambda q: f"Web results for: {q}",
            description="Search the web.",
        ),
    ]


@pytest.fixture(scope="function")
def memory_checkpointer():
    return MemorySaver()

