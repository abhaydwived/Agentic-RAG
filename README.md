# Enterprise Agentic RAG — Real-Time Multi-Tool Intelligence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Orchestration](https://img.shields.io/badge/Orchestrator-LangGraph%20v1.2-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LLM Provider](https://img.shields.io/badge/LLM-Google%20Gemini-20.svg)](https://ai.google.dev/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)


A state-of-the-art **Agentic Retrieval-Augmented Generation (RAG)** application built with **LangGraph**, **Google Gemini**, **Hybrid Vector + BM25 Search**, and **Real-Time Parallel Tool Execution**.

The system dynamically orchestrates queries across private documents (PDFs/TXTs), real-time stock financials, Wikipedia, live weather, and web search with automated anti-hallucination verification, hot-reload document indexing, and thread-isolated session memory.

---

## 📑 Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Project Objectives](#2-project-objectives)
- [3. Project Structure](#3-project-structure)
- [4. System Architecture](#4-system-architecture)
- [5. Agent Workflow & State Machine](#5-agent-workflow--state-machine)
- [6. Advanced RAG Pipeline](#6-advanced-rag-pipeline)
  - [Ingestion & Text Cleaning](#ingestion--text-cleaning)
  - [Recursive Chunking & Metadata Preservation](#recursive-chunking--metadata-preservation)
  - [Hybrid Retrieval (FAISS + BM25 via RRF)](#hybrid-retrieval-faiss--bm25-via-rrf)
  - [Cross-Encoder Neural Reranking](#cross-encoder-neural-reranking)
  - [Atomic Hot-Reload Re-Indexing](#atomic-hot-reload-re-indexing)
- [7. Google Gemini Integration & Rate-Limit Optimization](#7-google-gemini-integration--rate-limit-optimization)
- [8. Tool-Calling Mechanism & Resilience](#8-tool-calling-mechanism--resilience)
- [9. Conversation Memory & Session Isolation](#9-conversation-memory--session-isolation)
- [10. Observability, Evaluation & Benchmarking](#10-observability-evaluation--benchmarking)
- [11. Installation & Setup](#11-installation--setup)
- [12. Environment Variables](#12-environment-variables)
- [13. Running the Application](#13-running-the-application)
- [14. API Documentation](#14-api-documentation)
- [15. Multi-Tool Execution Examples](#15-multi-tool-execution-examples)
- [16. Limitations & Future Roadmap](#16-limitations--future-roadmap)

---

## 1. Problem Statement

Traditional naive RAG architectures suffer from critical flaws:
1. **Static Data Silos:** Pure vector databases cannot answer live questions (e.g. current stock prices, live weather, sports scores, today's news).
2. **Retrieval Blindspots:** Dense semantic embeddings miss exact technical keywords, IDs, or model codes, while sparse search (BM25) fails on synonyms and semantic intent.
3. **High Token Costs & Rate Limits:** Legacy architectures use 4 to 6 LLM calls for every user query (query understanding, routing, checking docs, generating response, self-critique), rapidly exhausting API quotas.
4. **Stale Vector Indexes on Document Upload:** In standard apps, uploading a new PDF requires taking the server down and manually rebuilding index files.
5. **Hallucination & Misattribution:** General-purpose LLMs hallucinate facts when private documents lack answers instead of stating missing coverage.
6. **Session Bleed:** Chat history often contaminates concurrent users without true thread-level checkpointer isolation.

---

## 2. Project Objectives

- **True Autonomous Agent:** Migrate from linear chains to a cyclic **LangGraph** workflow that plans, retrieves, executes tools, aggregates context, and validates output.
- **Google Gemini Integration:** Standardize on Google Gemini (`gemini-3.5-flash` / `gemini-2.5-flash`) via `langchain-google-genai` with fallback support for OpenAI (`gpt-4o-mini`).
- **Cost & Quota Efficiency:** Optimize from 5 LLM calls per query down to **exactly 1 LLM call per query** by utilizing deterministic rule-based query understanding, intent routing, and lexical response validation.
- **Hybrid Retrieval Fusion:** Combine dense **FAISS** vector embeddings with sparse **BM25** scores using **Reciprocal Rank Fusion (RRF)** and Cross-Encoder neural reranking.
- **Live Hot-Reload Uploads:** Enable instant retrieval from newly uploaded PDFs and TXTs without server restarts through atomic index swapping.
- **Thread-Isolated Memory:** Implement state checkpointing with LangGraph's `MemorySaver`.
- **100% CPU & Offline Capable:** Run all vector embeddings and reranking locally without GPU requirements.

---

## 3. Project Structure

```
Agentic RAG/
│
├── .env                              # Environment configuration (API keys, models, paths)
├── .env.example                      # Template for environment variables
├── requirements.txt                  # Production dependencies
├── README.md                         # Comprehensive project documentation
│
├── src/                              # Core Application Source Code
│   ├── __init__.py
│   ├── agent.py                      # LangGraph StateGraph builder, nodes, conditional edges
│   ├── graph_nodes.py                # High-efficiency agent node functions (0-LLM router, 1-LLM generator)
│   ├── graph_state.py                # TypedDict definition of AgentGraphState
│   ├── rag_pipeline.py               # Document loading, cleaning, chunking, FAISS, BM25, RRF, Reranking
│   ├── knowledge_indexer.py          # Atomic thread-safe FAISS indexer with hot-reload rebuild
│   ├── tool_registry.py              # Modular ToolRegistry managing RAG, Finance, Wiki, Weather, Web Search
│   ├── tool_executor.py              # Tool input validation, retries, timeouts, ThreadPool parallel execution
│   ├── session_manager.py            # LangGraph MemorySaver thread session manager
│   ├── evaluator.py                  # Evaluation layer: groundedness, citations, context relevance
│   ├── metrics.py                    # Recall@K, MRR, Context Relevance, Answer Faithfulness metrics
│   ├── tracer.py                     # Local execution tracer tracking latency, tools, and token metrics
│   ├── response_formatter.py         # Visual markdown formatting for UI and CLI responses
│   ├── server.py                     # FastAPI REST API (/chat, /upload, /health, /sessions)
│   ├── app_ui.py                     # Streamlit web UI with document uploader and multi-source separation
│   │
│   └── tools/                        # Real-Time Tool Integrations
│       ├── __init__.py
│       ├── rag_tool.py               # Hybrid RAG search tool
│       ├── finance_tool.py           # Yahoo Finance real-time stock and valuation tool
│       ├── wiki_tool.py              # MediaWiki REST API search (fast, robust User-Agent)
│       ├── weather_tool.py           # OpenWeatherMap current weather and forecast tool
│       └── web_search_tool.py        # Tavily real-time web search tool
│
├── tests/                            # Comprehensive Test Suite
│   ├── conftest.py                   # Pytest fixtures and mock dependencies
│   ├── test_all_nodes.py             # Unit tests for individual LangGraph nodes
│   ├── test_agent_graph.py           # End-to-end LangGraph state machine execution tests
│   ├── test_rag_pipeline.py          # Legacy RAG pipeline tests
│   ├── test_agent_tools_execution.py # Tool resilience, timeout, retry tests
│   ├── test_session_memory.py        # Checkpointer and session isolation tests
│   ├── test_server_endpoints.py      # Direct FastAPI route handler tests
│   ├── test_evaluation_layer.py      # Faithfulness and groundedness evaluator tests
│   ├── test_metrics_and_tracer.py    # Recall@K, MRR, tracer tests
│   │
│   └── phase8/                       # Phase 8 Pytest Test Suite (78 Tests)
│       ├── test_ingestion.py         # Ingestion, normalization, PDF/TXT loaders
│       ├── test_chunking_embedding.py# Recursive chunking, metadata preservation, CPU embeddings
│       ├── test_retrieval.py         # BM25, FAISS, Hybrid RRF, Reranker, context assembly
│       ├── test_tool_execution.py    # Validation, retries, timeouts, parallelism, routing
│       └── test_memory_and_api.py    # Thread isolation, session resets, FastAPI endpoints
│
├── scripts/                          # Evaluation and Automation Scripts
│   └── run_evaluation.py             # Reproducible evaluation runner for Recall@K, MRR, Faithfulness
│
└── data/                             # Data and Knowledge Storage
    ├── eval_dataset.json             # Ground-truth evaluation dataset with queries and golden answers
    ├── eval_results.json             # Benchmark outputs and latency metrics
    │
    └── knowledge_base/               # Uploaded Documents Storage
        ├── .gitkeep
        ├── employee_handbook.txt     # Sample company policies document
        ├── financial_report.txt      # Sample quarterly financial targets document
        ├── Profile.pdf               # Sample uploaded resume/profile document
        ├── Week 11.pdf               # Sample uploaded multi-page presentation document
        ├── Chebyshev's Method in Python.pdf
        └── A WBC Framework for Human-Like Walking with Knee Stretch on Flat-Foot Humanoids.pdf
```

---

## 4. System Architecture

```
                                 ┌────────────────────────┐
                                 │      Client Layer      │
                                 │ (Streamlit Web UI /    │
                                 │   REST API Consumers)  │
                                 └───────────┬────────────┘
                                             │ HTTP
                                             ▼
                                 ┌────────────────────────┐
                                 │     FastAPI Backend    │
                                 │      (Port: 8000)      │
                                 └───────────┬────────────┘
                                             │ thread_id
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │         LangGraph Agent Runtime         │
                        │                                         │
                        │  [Query Understanding & Normalization]  │
                        │                    │                    │
                        │       [Intent Classification]           │
                        │                    │                    │
                        │         ┌──────────┴──────────┐         │
                        │         ▼                     ▼         │
                        │  [Internal Hybrid     [External Tool    │
                        │     RAG Engine]          Execution]     │
                        │         │                     │         │
                        │         └──────────┬──────────┘         │
                        │                    ▼                    │
                        │        [Context Aggregation &           │
                        │         Source Attribution]             │
                        │                    │                    │
                        │      [Grounded Answer Generation]       │
                        │         (Google Gemini API)             │
                        │                    │                    │
                        │       [Response Validation Guard]       │
                        └────────────────────┬────────────────────┘
                                             │
                   ┌─────────────────────────┴─────────────────────────┐
                   ▼                                                   ▼
      ┌────────────────────────┐                          ┌────────────────────────┐
      │   Private Knowledge    │                          │   Live External APIs   │
      │         Base           │                          ├────────────────────────┤
      ├────────────────────────┤                          │ • Yahoo Finance        │
      │ • PDF / TXT Ingestion  │                          │ • OpenWeatherMap       │
      │ • all-MiniLM-L6-v2     │                          │ • MediaWiki REST       │
      │ • FAISS Vector Store   │                          │ • Tavily Web Search    │
      │ • Rank-BM25 Index      │                          └────────────────────────┘
      └────────────────────────┘
```

---

## 5. Agent Workflow & State Machine

```
   (START)
      │
      ▼
[query_understanding]   ──► Resolves follow-up pronouns (it, its, he, ticker) and extracts entities (0 LLM calls)
      │
      ▼
[intent_classification]  ──► Fast regex & keyword rule-based routing to KB vs External Tools (0 LLM calls)
      │
      ├───────────────────────┬───────────────────────┐
      ▼                       ▼                       ▼
[document_retrieval]    [external_tools]       [answer_generation] (Direct Q&A)
      │                       │                       │
      ├───────────────────────┘                       │
      ▼                                               │
[context_aggregation]  ──► Formats structured sections │
      │                   with [1] Source: doc.pdf    │
      ▼                                               │
[answer_generation]  <────────────────────────────────┘ (Exactly 1 Google Gemini API call)
      │
      ▼
[response_validation]  ──► Rule-based confidence check (0 LLM calls)
      │
      ├─► (Valid / Max Iterations) ──► (END)
      └─► (Needs Revision)         ──► Loops back to [answer_generation]
```

---

## 6. Advanced RAG Pipeline

Located in [`src/rag_pipeline.py`](src/rag_pipeline.py) and [`src/knowledge_indexer.py`](src/knowledge_indexer.py).

### Ingestion & Text Cleaning
- **Directory Traversal:** Automatically indexes all `.pdf` and `.txt` files in `data/knowledge_base/`.
- **Normalization:** Strips null bytes (`\x00`), resolves hyphenated line wraps (`algo-\nrithm` $\rightarrow$ `algorithm`), normalizes non-breaking whitespace (`\xa0`), and collapses redundant spacing.
- **PDF Parsing:** Per-page PDF extraction using `pypdf`, preserving exact page numbers for citations.

### Recursive Chunking & Metadata Preservation
- Implements `RecursiveCharacterTextSplitter` with `chunk_size=500` and `chunk_overlap=50`.
- Chunks preserve structural metadata:
  ```python
  {
      "chunk_id": "Week 11.pdf#p120_c130",
      "chunk_index": 130,
      "page": 120,
      "document_name": "Week 11.pdf",
      "source": "data/knowledge_base/Week 11.pdf",
      "file_type": "pdf"
  }
  ```

### Hybrid Retrieval (FAISS + BM25 via RRF)
Combines dense semantic similarity and sparse keyword matching using **Reciprocal Rank Fusion (RRF)**:
```math
$$\text{RRF\_Score}(d) = \frac{0.60}{60 + \text{Rank}_{\text{dense}}(d)} + \frac{0.40}{60 + \text{Rank}_{\text{bm25}}(d)}$$
```

- **Dense Retriever:** Local `sentence-transformers/all-MiniLM-L6-v2` dense vectors via `FAISS`.
- **Sparse Retriever:** Exact token match via `rank_bm25.BM25Okapi`.

### Cross-Encoder Neural Reranking
Optional neural reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2` scoring $(query, passage)$ pairs directly. Operates on CPU or gracefully falls back to top RRF candidates when offline.

### Atomic Hot-Reload Re-Indexing
When a user uploads a new document:
1. File is saved to `data/knowledge_base/`.
2. The indexer writes the new FAISS index to a temporary directory (`.faiss_tmp_*`).
3. Replaces `.faiss_index` **atomically** under a thread lock (`_REBUILD_LOCK`).
4. Updates `_GLOBAL_PIPELINE` in-place so all running queries immediately search the new file without restarting the server.

---

## 7. Google Gemini Integration & Rate-Limit Optimization

The application is powered by Google Gemini via `langchain-google-genai`:

- **Primary Model:** `gemini-3.5-flash` (or `gemini-2.5-flash` via `.env`).
- **Quota Challenge:** The Gemini free tier imposes a strict daily limit of **20 requests per day**.
- **Solution:** In legacy architectures, a single query made 5 sequential LLM calls:
  $$\text{Query Understanding} \rightarrow \text{Intent Routing} \rightarrow \text{Context Evaluation} \rightarrow \text{Answer Generation} \rightarrow \text{Validation}$$
  This exhausted the daily quota in just 4 questions.
- **Optimization:** We redesigned nodes in [`src/graph_nodes.py`](src/graph_nodes.py):
  - **Query Understanding:** Deterministic regex & follow-up resolver (**0 LLM calls**).
  - **Intent Classification:** Keyword/entity matching (**0 LLM calls**).
  - **Document Retrieval:** Local FAISS + BM25 (**0 LLM calls**).
  - **Answer Generation:** Dedicated factual generation with anti-hallucination prompt (**1 LLM call**).
  - **Response Validation:** Lexical completeness check (**0 LLM calls**).
- **Outcome:** **Exactly 1 LLM call per user query**, allowing up to 20 full multi-turn conversations daily on the free tier.

---

## 8. Tool-Calling Mechanism & Resilience

Located in [`src/tool_executor.py`](src/tool_executor.py) and [`src/tool_registry.py`](src/tool_registry.py).

- **Input Validation:** Rejects empty queries, script injection (`<script>`), URL schemes (`http://`), and oversized inputs.
- **Transient Error Retry:** Automatically retries network socket drops with exponential backoff.
- **Execution Timeouts:** Terminates unresponsive external requests after 5.0 seconds.
- **Parallel Multi-Tool Execution:** Independent tool calls execute concurrently via `ThreadPoolExecutor(max_workers=5)`.
- **Integrated Tool Suite:**
  - `search_knowledge_base`: Hybrid RAG over uploaded PDFs/TXTs.
  - `get_stock_data`: Real-time stock quotes, PE ratios, and market caps via `yfinance`.
  - `search_wikipedia`: Fast MediaWiki REST API search with custom User-Agent headers.
  - `get_weather`: Current temperature, humidity, and weather conditions via `OpenWeatherMap`.
  - `web_search`: Live web search results via `Tavily`.

---

## 9. Conversation Memory & Session Isolation

Located in [`src/session_manager.py`](src/session_manager.py).

- Backed by LangGraph's native `MemorySaver` checkpointer.
- Each user session has a unique `thread_id` UUID.
- **Session Isolation:** History in Session A is completely isolated and never leaks into Session B.
- **Pronoun Resolution:** The agent understands contextual follow-ups like *"What is its market cap?"* or *"What did he work on before Addverb?"*.
- **Session Reset:** Dedicated `DELETE /sessions/{session_id}` endpoint clears memory and issues a clean session thread instantly.

---

## 10. Observability, Evaluation & Benchmarking

Located in [`src/evaluator.py`](src/evaluator.py), [`src/metrics.py`](src/metrics.py), and [`scripts/run_evaluation.py`](scripts/run_evaluation.py).

Tracks live execution metrics:
- **Recall@K & MRR:** Measures retrieval coverage against `data/eval_dataset.json`.
- **Context Relevance:** Measures the fraction of retrieved passages directly relevant to the query.
- **Answer Faithfulness:** Detects hallucinations by verifying claims against retrieved document sentences.
- **Latency & Tool Tracking:** [`src/tracer.py`](src/tracer.py) logs execution duration, tool names, and iteration counts.

Run the reproducible benchmark:
```powershell
.\.venv\Scripts\python.exe scripts/run_evaluation.py
```

---

## 11. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Windows / Linux / macOS)
- Git

### Setup Steps

```powershell
# 1. Clone repository
git clone https://github.com/abhaydwived/Agentic-RAG
cd "Agentic-RAG"

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Linux/macOS: source .venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install test dependencies
pip install pytest pytest-asyncio httpx
```

---

## 12. Environment Variables

Configure your `.env` file in the root folder:

```ini
# Google Gemini Configuration
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.5-flash
GEMINI_API_KEY=AIzaSy...your_gemini_key_here

# Alternative: OpenAI Configuration
# LLM_PROVIDER=openai
# OPENAI_MODEL=gpt-4o-mini
# OPENAI_API_KEY=sk-...your_openai_key_here

# External Real-Time Tools
TAVILY_API_KEY=tvly-...your_tavily_key_here
OPENWEATHERMAP_API_KEY=your_openweathermap_key_here

# System Settings
KNOWLEDGE_BASE_DIR=data/knowledge_base
BACKEND_URL=http://127.0.0.1:8000
```

---

## 13. Running the Application

### 1. Run the Test Suite (78 Tests — 100% Pass)
```powershell
.\.venv\Scripts\python.exe -m pytest tests/phase8 -v
```

### 2. Start the FastAPI Backend
```powershell
.\.venv\Scripts\python.exe -m uvicorn src.server:app --host 127.0.0.1 --port 8000 --reload
```
Swagger API docs: **http://127.0.0.1:8000/docs**

### 3. Start the Streamlit Web Interface
```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app_ui.py
```
Browser interface: **http://localhost:8501**

---

## 14. API Documentation

### `POST /chat`
Submits a question to the agent.

**Request:**
```json
{
  "query": "What are the remote work rules in our handbook, and what is the price of NVDA?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "session_id": "902d2959-1588-43d9-a78b-d72b21cf56b8",
  "query": "What are the remote work rules in our handbook, and what is the price of NVDA?",
  "answer": "According to the Employee Handbook, full-time employees may work remotely up to 3 days per week with manager approval. In live financial markets, NVIDIA (NVDA) is trading at $138.25.",
  "tools_used": [
    "search_knowledge_base",
    "get_stock_data"
  ],
  "retrieved_sources": "[1] Source: employee_handbook.txt (Page 1 | Chunk ID: employee_handbook.txt#p1_c1 | Score: 0.965)\nRemote Work Policy: Eligible team members may work remotely up to three days per week...",
  "live_external_data": {
    "get_stock_data": "Ticker: NVDA, Price: $138.25, Change: +2.4%, Market Cap: $3.39T"
  },
  "iteration_count": 1,
  "is_valid": true
}
```

---

### `POST /upload`
Uploads and indexes a `.pdf` or `.txt` file with atomic hot-reload.

**Request:** `multipart/form-data` with `file=@MyDocument.pdf`

**Response:**
```json
{
  "status": "success",
  "filename": "MyDocument.pdf",
  "message": "'MyDocument.pdf' uploaded and indexed. Knowledge base now has 156 chunks from 4 document(s).",
  "documents_in_kb": [
    "MyDocument.pdf",
    "Profile.pdf",
    "employee_handbook.txt",
    "financial_report.txt"
  ]
}
```

---

### `GET /health`
Exposes system status, active LLM model, registered tools, and indexed documents.

### `GET /sessions`
Lists all active conversation session identifiers.

### `DELETE /sessions/{session_id}`
Clears session history and returns a clean conversation thread ID.

---

## 15. Multi-Tool Execution Examples

### Example 1: Uploaded PDF Query + Real-Time Financials
- **User Query:** *"What skills are listed in Rajesh's profile, and how is Apple stock doing today?"*
- **Execution Plan:** `['search_knowledge_base', 'get_stock_data']`
- **Retrieval:** Extracts skills (Software Design, Path Planning, OpenCV, MPC) from `Profile.pdf`.
- **Live Tool:** Fetches real-time price for `AAPL` via `yfinance`.
- **Synthesis:** Produces a dual-part verified response with exact citations and live stock stats.

### Example 2: General Knowledge & Weather Query
- **User Query:** *"Who was the captain of the 1983 World Cup winning team and what is the weather in Delhi?"*
- **Execution Plan:** `['search_wikipedia', 'get_weather']`
- **Wikipedia:** Retrieves Kapil Dev's captaincy from Wikipedia REST API.
- **Weather:** Queries OpenWeatherMap for Delhi's current temperature.
- **Synthesis:** Answers both queries clearly separated in the UI.

---

## 16. Limitations & Future Roadmap

### Limitations
1. **CPU Embedding Build Time on Large Files:** Ingesting 200+ page PDFs on CPU takes 30–45 seconds for vector indexing.
2. **Scanned PDF Text:** Current parser requires selectable text in PDFs; scanned image PDFs need an OCR pre-processor.

### Future Roadmap
- [ ] **Incremental Vector Appending:** Add incremental vector insertions (`add_documents`) instead of rebuilding full FAISS index on upload.
- [ ] **OCR Ingestion:** Integrate `pytesseract` or Google Cloud Vision OCR for scanned image PDFs.
- [ ] **Persistent Vector Database:** Add optional connectors for Pinecone, Qdrant, and Milvus alongside local FAISS.
- [ ] **Streaming Responses:** Support Server-Sent Events (SSE) token streaming in `/chat`.

---

## 📄 License

Apache 2.0 License. Free for commercial and research use.
